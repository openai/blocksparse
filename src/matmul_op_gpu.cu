
#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>
#include <type_traits>

template <typename V>
__global__ void __launch_bounds__(128) gemm_32x32x32_TN_vec4(float* U, const V* __restrict__ X, const V* __restrict__ E, uint C, uint K, uint N, uint C16, uint K16, uint inc_n, uint inc_c, uint inc_k)
{
    __shared__ float shrU[32*32*2 + 16*4];

    uint tid   = threadIdx.x;
    uint idx_C = blockIdx.y;
    uint idx_K = blockIdx.x;
    uint idx_N = blockIdx.z;

    uint tx = tid  & 7;
    uint ty = tid >> 3;
    uint  n = idx_N*32 + ty;

    // global offsets in vector units
    uint c = idx_C*8 + tx;
    uint k = idx_K*8 + tx;
    uint offsetC = n*C + c;
    uint offsetK = n*K + k;

    bool bc = c < C;
    bool bk = k < K;

    // shared offsets in bytes
    // When reading, each warp works on its own 8 rows.
    // These groups of 8 are added together at end.
    uint writeS = (ty*32 + tx*4) * 4;
    uint row8   = (tid & 96) * 32;
    uint readCs = row8 + (((tid & 16) >> 3) | (tid & 1)) * 16;
    uint readKs = row8 + ((tid >> 1) & 7) * 16;

    // This keeps all prior logic outside of the loops.
    asm("mov.b32 %0, %0;" : "+r"(writeS)  : );
    asm("mov.b32 %0, %0;" : "+r"(offsetC) : );
    asm("mov.b32 %0, %0;" : "+r"(offsetK) : );
    asm("mov.b32 %0, %0;" : "+r"(readCs)  : );
    asm("mov.b32 %0, %0;" : "+r"(readKs)  : );

    // zero 32 accumulation registers
    float regU[8][4]; // [c][k]
    for (int c = 0; c < 8; c++)
        for (int k = 0; k < 4; k++)
            regU[c][k] = 0;

    // assume a minimum of one loop
    #pragma unroll 1
    do
    {
        V c00, c16;
        V k00, k16;
        ew_zero(c00); ew_zero(c16);
        ew_zero(k00); ew_zero(k16);
        const V* X00 = add_ptr_u(X, offsetC +   0);
        const V* X16 = add_ptr_u(X, offsetC + C16);
        const V* E00 = add_ptr_u(E, offsetK +   0);
        const V* E16 = add_ptr_u(E, offsetK + K16);
        if (bc)
        {
            c00 = __ldg(X00);
            c16 = __ldg(X16);
        }
        if (bk)
        {
            k00 = __ldg(E00);
            k16 = __ldg(E16);
        }
        offsetC += inc_c;
        offsetK += inc_k;
        n       += inc_n;

        __syncthreads();
        st_shared_v4(writeS + ( 0*32 + 0*16*32)*4, to_float(c00));
        st_shared_v4(writeS + ( 0*32 + 1*16*32)*4, to_float(c16));
        st_shared_v4(writeS + (32*32 + 0*16*32)*4, to_float(k00));
        st_shared_v4(writeS + (32*32 + 1*16*32)*4, to_float(k16));
        __syncthreads();

        float regC[8], regK[4];

        #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            // fetch outer product data
            ld_shared_v4(readCs + ( 0*32 + 32*j +  0)*4, &regC[0] );
            ld_shared_v4(readCs + ( 0*32 + 32*j + 16)*4, &regC[4] );
            ld_shared_v4(readKs + (32*32 + 32*j +  0)*4,  regK    );
            // compute outer product
            for (int c = 0; c < 8; c++)
                for (int k = 0; k < 4; k++)
                    regU[c][k] += regC[c] * regK[k];
        }
    } while (n < N);

    // conserve registers by forcing a reload of these
    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  ) :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_K) :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_C) :);

    // Arrange 4 tiles horizontally in the X direction: ((tid & 96) >> 2)
    // Add some spacing  to avoid write bank conflicts: (tidY << 2)
    int tidY = ((tid & 16) >> 3) | (tid & 1);
    int tidX = ((tid >> 1) & 7) + ((tid & 96) >> 2) + (tidY << 2);

    float4* storU4 = (float4*)&shrU[tidY*32*4*4 + tidX*4];

    __syncthreads();

    storU4[0*8*4] = *(float4*)regU[0];
    storU4[1*8*4] = *(float4*)regU[1];
    storU4[2*8*4] = *(float4*)regU[2];
    storU4[3*8*4] = *(float4*)regU[3];

    __syncthreads();

    // leaving vector math
    uint tid31 = tid & 31;
    uint tid32 = tid >> 5;
    C *= 4;
    K *= 4;

    float* readU = &shrU[tid32*32*4 + tid31];

    float u[4][4];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            u[j][i] = readU[j*32*4*4 + j*16 + i*32];

    // Tree reduce
    for (int k = 0; k < 4; k++)
        for (int j = 2; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                u[k][i] += u[k][i+j];

    k = idx_K*32 + tid31;
    c = idx_C*32 + tid32;
    bk = k < K;

    uint offsetU = c*K + k;
    atomicRed(add_ptr_u(U, offsetU +  0*K), u[0][0], 0, bk && c +  0 < C);
    atomicRed(add_ptr_u(U, offsetU +  4*K), u[1][0], 0, bk && c +  4 < C);
    atomicRed(add_ptr_u(U, offsetU +  8*K), u[2][0], 0, bk && c +  8 < C);
    atomicRed(add_ptr_u(U, offsetU + 12*K), u[3][0], 0, bk && c + 12 < C);

    __syncthreads();

    storU4[0*8*4] = *(float4*)regU[4];
    storU4[1*8*4] = *(float4*)regU[5];
    storU4[2*8*4] = *(float4*)regU[6];
    storU4[3*8*4] = *(float4*)regU[7];

    __syncthreads();

    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            u[j][i] = readU[j*32*4*4 + j*16 + i*32];

    // Tree reduce
    for (int k = 0; k < 4; k++)
        for (int j = 2; j > 0; j >>= 1)
            for (int i = 0; i < j; i++)
                u[k][i] += u[k][i+j];

    atomicRed(add_ptr_u(U, offsetU + 16*K), u[0][0], 0, bk && c + 16 < C);
    atomicRed(add_ptr_u(U, offsetU + 20*K), u[1][0], 0, bk && c + 20 < C);
    atomicRed(add_ptr_u(U, offsetU + 24*K), u[2][0], 0, bk && c + 24 < C);
    atomicRed(add_ptr_u(U, offsetU + 28*K), u[3][0], 0, bk && c + 28 < C);
}

# if __CUDA_ARCH__ >= 700

#include <mma.h>
using namespace nvcuda::wmma;

extern "C"
__device__ __noinline__ void output_gemm_64x64x32_TN(float* fShare, float* U, uint C, uint K, uint offsetU, uint readU)
{
    for (int i = 0; i < 8; i++)
        atomicRed(U + (offsetU + i*K), fShare[readU + i*272] + fShare[readU + i*272 + 128]);
}

extern "C"
__global__ void __launch_bounds__(256) hmma_gemm_64x64x32_TN_vec8(float* U, const ehalf8* __restrict__ X, const ehalf8* __restrict__ E, uint C, uint K, uint N, uint inc_n, uint inc_c, uint inc_k)
{
    __shared__  float fShare[(256+16)*16];
    half* hShare = (half*)&fShare[0];

    uint tid   = threadIdx.x;
    uint idx_C = blockIdx.y;
    uint idx_K = blockIdx.x;
    uint idx_N = blockIdx.z;
    uint tid31 = tid & 31;

    uint tx = tid  & 7;
    uint ty = tid >> 3;
    uint  n = idx_N*32 + ty;

    // global offsets in vector units
    uint c = idx_C*8 + tx;
    uint k = idx_K*8 + tx;
    uint offsetC = n*C + c;
    uint offsetK = n*K + k;

    // bool bc = c < C;
    // bool bk = k < K;
    asm volatile (".reg .pred bc; setp.lt.u32 bc, %0, %1;" :: "r"(c), "r"(C));
    asm volatile (".reg .pred bk; setp.lt.u32 bk, %0, %1;" :: "r"(k), "r"(K));

    // When reading, each group of 4 warp works on its own 16 rows.
    // These 2 groups of 16 are added together at end.
    // Also add 16 elements per row to reduce bank conflicts.
    uint writeS = ty*80*2 + tx*8*2; // byte units
    uint row16  = (tid & 128) * (16*80/128);
    uint readCs = row16 + (tid & 64) * 32/64 + (tid31 & 3)*80 + (tid31 & 4)*2 + (tid31 & 16)/4;
    uint readKs = row16 + (tid & 32) + 32*80 + (tid31 & 3)*80 + (tid31 & 8)   + (tid31 & 16)/4;

    fragment<accumulator,16,16,16,float> fragU[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            fill_fragment(fragU[i][j], 0.0f);

    // assume a minimum of one loop
    #pragma unroll 1
    do
    {
        asm volatile ("{\n\t"
            ".reg .u32 c<4>, k<4>;\n\t"
            "mov.u32 c0, 0;\n\t"
            "mov.u32 c1, 0;\n\t"
            "mov.u32 c2, 0;\n\t"
            "mov.u32 c3, 0;\n\t"
            "mov.u32 k0, 0;\n\t"
            "mov.u32 k1, 0;\n\t"
            "mov.u32 k2, 0;\n\t"
            "mov.u32 k3, 0;\n\t"
            "@bc ld.global.nc.v4.u32 {c0, c1, c2, c3}, [%0];\n\t"
            "@bk ld.global.nc.v4.u32 {k0, k1, k2, k3}, [%1];\n\t"
            "bar.sync 0;\n\t"
            "st.shared.v4.u32 [%2 +  0*80*2], {c0, c1, c2, c3};\n\t"
            "st.shared.v4.u32 [%2 + 32*80*2], {k0, k1, k2, k3};\n\t"
            "bar.sync 0;\n\t"
            "}" :: "l"(X + offsetC), "l"(E + offsetK), "r"(writeS));
        offsetC += inc_c;
        offsetK += inc_k;
        n       += inc_n;

        fragment<matrix_a,16,16,16,half,col_major> fragC[2];
        fragment<matrix_b,16,16,16,half,row_major> fragK[2];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
            {
                *(uint2*)&fragC[i].x[j*4] = *(uint2*)&hShare[readCs + i*16 + j*4*80];
                *(uint2*)&fragK[i].x[j*4] = *(uint2*)&hShare[readKs + i*16 + j*4*80];
            }

        mma_sync(fragU[0][0], fragC[0], fragK[0], fragU[0][0], false);
        mma_sync(fragU[1][0], fragC[1], fragK[0], fragU[1][0], false);
        mma_sync(fragU[1][1], fragC[1], fragK[1], fragU[1][1], false);
        mma_sync(fragU[0][1], fragC[0], fragK[1], fragU[0][1], false);

    } while (n < N);

    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  ) :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_K) :);
    asm volatile ("mov.u32 %0, %ctaid.y;" : "=r"(idx_C) :);

    uint storU = (tid & 224) + ((tid & 1) + (tid & 4)*2 + (tid & 16)/4)*272 + (tid & 2) + (tid & 8);
    uint readU = (tid & 127) + (tid & 128) * (272*8/128);

    // leaving vector math
    C *= 8;
    K *= 8;
    k = idx_K*64 + (tid & 63);
    c = idx_C*64 + (tid & 64)*32/64 + (tid & 128)*8/128;
    bool bk = k < K;
    uint offsetU = c*K + k;

    #pragma unroll
    for (int i = 0; i < 2; i++)
    {
        __syncthreads();
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                for (int m = 0; m < 2; m++)
                    *(float2*)&fShare[storU + j*16 + k*4 + m*2*272] = *(float2*)&fragU[i][j].x[k*4 + m*2];
            //store_matrix_sync(&fShare[storU + j*16], fragU[i][j], 272, mem_row_major);
        __syncthreads();

        if (c + i*16 < C && bk)
            output_gemm_64x64x32_TN(fShare, U, C, K, offsetU + i*16*K, readU);
    }
}

#else // __CUDA_ARCH__ >= 700

__global__ void __launch_bounds__(256) hmma_gemm_64x64x32_TN_vec8(float* U, const ehalf8* __restrict__ X, const ehalf8* __restrict__ E, uint C, uint K, uint N, uint inc_n, uint inc_c, uint inc_k)
{
    *U = 0;
}

#endif // __CUDA_ARCH__ >= 700

template <typename V>
bool Gemm_TN(CUstream stream, uint SMs, int major,
          float* u,
    const     V* x,
    const     V* e,
    uint C, uint K, uint N)
{
    cuMemsetD32Async((CUdeviceptr)u, 0, C*K, stream);

    if (std::is_same<V, ehalf4>::value && major >= 7 && (C & 7) == 0 && (K & 7) == 0)
    {
        const ehalf8* X = (const ehalf8*)x;
        const ehalf8* E = (const ehalf8*)e;

        uint gridK = CEIL_DIV(K, 64);
        uint gridC = CEIL_DIV(C, 64);
        uint gridN = CEIL_DIV(N, 32);
        C >>= 3;
        K >>= 3;

        // target 4 blocks per SM
        uint segments = SMs, tiles = gridK*gridC;
             if (tiles >= 64) segments /= 8;
        else if (tiles >= 16) segments /= 4;
        else if (tiles >   4) segments /= 2;
        else if (tiles ==  2) segments *= 2;
        else if (tiles ==  1) segments *= 4;

        if (segments > gridN)
            segments = gridN;
        uint seg_len = segments*32;

        dim3 grid(gridK, gridC, segments);
        hmma_gemm_64x64x32_TN_vec8<<<grid,256,0,stream>>>(u, X, E, C, K, N, seg_len, seg_len*C, seg_len*K);
        return true; // TODO
    }

    uint gridK = CEIL_DIV(K, 32);
    uint gridC = CEIL_DIV(C, 32);
    uint gridN = CEIL_DIV(N, 32);
    C >>= 2;
    K >>= 2;

    // target mult of 6 blocks per SM
    uint smMult = 1, tiles = gridK*gridC;
         if (tiles == 1) smMult = 6;
    else if (tiles <= 4) smMult = 3;
    uint segments = SMs*smMult;
    if (segments > gridN)
        segments = gridN;
    uint seg_len = segments*32;

    dim3 grid(gridK, gridC, segments);
    gemm_32x32x32_TN_vec4<V><<<grid,128,0,stream>>>(u, x, e, C, K, N, C*16, K*16, seg_len, seg_len*C, seg_len*K);
    return true; // TODO
}

template bool Gemm_TN<float4>(CUstream stream, uint SMs, int major, float* u, const float4* x, const float4* e, uint C, uint K, uint N);
template bool Gemm_TN<ehalf4>(CUstream stream, uint SMs, int major, float* u, const ehalf4* x, const ehalf4* e, uint C, uint K, uint N);





#endif // GOOGLE_CUDA
