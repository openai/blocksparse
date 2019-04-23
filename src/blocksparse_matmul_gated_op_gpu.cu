
#if GOOGLE_CUDA

#include "ew_op_gpu.h"
#include <stdio.h>

typedef struct __align__(16) LutEntry
{
    int offsetX;
    int offsetW;
    float gate;
    float unused;
} LutEntry;


template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(32) gemm_blocksparse_gated_08x64x08x8_xprop(
    const  int2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N /* N is in units of groups of 8 elements each (N/8) */)
{
    if (Fprop)
        asm(".shared .align 16 .b32 share[576];" ::); // 576 =  8*8 + 64*8
    else
        asm(".shared .align 16 .b32 share[608];" ::); // 608 = 12*8 + 64*8

    extern __shared__ LutEntry Lut4_s[];
    LutEntry* Lut4s = &Lut4_s[Fprop ? 576/4 : 608/4];

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

    uint dep_thd_mask = 0xffffffff; dep_thd_mask >>= 32 - tid;

    int new_lut_size = 0;

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
    {
        LutEntry entry;
        *(int2*)&entry = Lut[i];

        entry.gate = Gate[entry.offsetW];

        // only add the entry to the lut if the gate is non-zero
        // compiler is stupid about reusing predicate here so use asm
        uint ballot, warp_non_zero;
        asm volatile ("{\n\t"
            ".reg .pred p;                      \n\t"
            ".reg .u32 ballot;                  \n\t"
            "setp.ne.ftz.f32 p, %2, 0f00000000; \n\t"
# if CUDA_VERSION >= 9020
            "vote.sync.ballot.b32 ballot, p, 0xffffffff;\n\t"
# else
            "vote.ballot.b32 ballot, p;         \n\t"
# endif
            "mov.b32  %0, ballot;               \n\t"
            "popc.b32 %1, ballot;               \n\t"
            "@!p bra GATE_ZERO;                 \n\t"
            "}" : "=r"(ballot), "=r"(warp_non_zero) : "f"(entry.gate));
        {
            uint dep_thd_cnt = __popc(dep_thd_mask & ballot);

            entry.unused   = 0;
            entry.offsetX *= N8;    // 8 lines of N per block
            entry.offsetW *= 64*2;  // 64 entries of W per block, 2 bytes each
            Lut4s[new_lut_size + dep_thd_cnt] = entry;
        }
        asm volatile ("\nGATE_ZERO:\n" ::);

        new_lut_size += warp_non_zero;
    }
    //lut_size = new_lut_size;
# if CUDA_VERSION >= 9020
    asm volatile ("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;" : "=r"(lut_size) : "r"(new_lut_size));
# else
    asm volatile ("shfl.idx.b32 %0, %1, 0, 0x1f;" : "=r"(lut_size) : "r"(new_lut_size));
# endif

    // zero accumulation registers
    float regY[4][8];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 8; x++)
            // use asm here to ensure this happens after lut table construction and before main loop skipping
            asm volatile ("mov.b32 %0, 0;" : "=f"(regY[w][x]) :);

    // skip loop if empty lut
    // Compiler generates suboptimal code if a simple "for loop" is used.
    asm volatile (".reg .pred lut_zero; \n\t"
        "setp.eq.u32 lut_zero, %0, 0;   \n\t"
        "@lut_zero bra.uni END_LOOP;" :: "r"(lut_size));

    int i = 0; do
    {
        LutEntry entry = Lut4s[i++];

        entry.offsetX += offsetX;
        entry.offsetW += tid*4;

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
            "}" : "=l"(W0),"=l"(X0),"=l"(X4) : "l"(W), "l"(X), "r"(entry.offsetW), "r"(entry.offsetX), "r"(entry.offsetX + N*4*8*2) );

        // Fetch 8 rows at a time from W and X
        float2 w0 = load(W0, 0);
        float8 x0 = load(X0, 0, bn);
        float8 x4 = load(X4, 0, bn);

        w0 = ew_mul(w0, entry.gate);

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

    }  while (i < lut_size);

    asm volatile ("\nEND_LOOP:\n":: );
    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  ) :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_N) :);

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
            outY[w][x] += shfl_xor(swap, 16);
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
        __syncwarp();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;
        __syncwarp();

        if (count == 0)
        {
            if (tid == 0)
                *Count = 1;

            // first block to get here just writes out to init the memory
            store(Y, *(float8*)outY[0], N*0, bn);
            store(Y, *(float8*)outY[1], N*2, bn);

            __threadfence();
            __syncwarp();

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
            __syncwarp();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}


template <bool Fprop, typename TW, typename TX, typename TY>
__global__ void __launch_bounds__(32) gemm_blocksparse_gated_08x64x08x4_xprop(
    const  int2* __restrict__ Lut,
    const float* __restrict__ Gate,
    const    TW* __restrict__ W,
    const    TX* __restrict__ X,
    TY* Y, int* Lock, int locks, int N)
{
    __shared__ float shrX1[64*8];
    __shared__ float shrW1[Fprop ? 8*8 : 12*8];
   float2* shrW2 = (float2*)shrW1;
   float2* shrX2 = (float2*)shrX1;
   float4* shrX4 = (float4*)shrX1;

    extern __shared__ LutEntry Lut4s[];

    int tid   = threadIdx.x;
    int idx_N = blockIdx.x;
    int idx_L = blockIdx.y;

    int4 lut_head = ((const int4*)Lut)[idx_L];

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
    //printf("%d %2d %d\n", idx_K, tid, lut_size);

    uint dep_thd_mask = 0xffffffff; dep_thd_mask >>= 32 - tid;
    int  new_lut_size = 0;

    // prefetch the lut data into shared
    Lut += lut_offset;
    #pragma unroll 1
    for (int i = tid; i < lut_size; i += 32)
    {
        //printf("%d %2d %d %d %d\n", idx_K, tid, i, lut_size, new_lut_size);

        LutEntry entry;
        *(int2*)&entry = Lut[i];

        entry.gate = Gate[entry.offsetW];

        // only add the entry to the lut if the gate is non-zero
        bool gate_non_zero = entry.gate != 0.0f;
        //uint gate_ballot   = __ballot_sync(0xffffffff, gate_non_zero);
        uint gate_ballot   = __ballot(gate_non_zero);
        uint warp_non_zero = __popc(gate_ballot);
        if (gate_non_zero)
        {
            uint dep_thd_cnt = __popc(dep_thd_mask & gate_ballot);

            entry.unused   = 0;
            entry.offsetX *= N8;
            entry.offsetW *= 32;
            Lut4s[new_lut_size + dep_thd_cnt] = entry;
        }
        new_lut_size += warp_non_zero;
    }
    // lut_size = new_lut_size;
// # if CUDA_VERSION >= 9020
//     lut_size = __shfl_sync(0xffffffff, new_lut_size, 0, 32);
// # else
    lut_size = __shfl(new_lut_size, 0, 32);
// # endif

    //printf("%d %2d %d\n", idx_K, tid, lut_size);

    // zero accumulation registers
    float regY[4][4];
    for (int w = 0; w < 4; w++)
        for (int x = 0; x < 4; x++)
            asm volatile ("mov.b32 %0, 0;" : "=f"(regY[w][x]) :);

    // skip loop if empty lut
    // Compiler generates suboptimal code if a simple "for loop" is used.
    asm volatile (".reg .pred lut_zero; \n\t"
        "setp.eq.u32 lut_zero, %0, 0;   \n\t"
        "@lut_zero bra.uni END_LOOP;" :: "r"(lut_size));

    // loop over each lut entry to compute a gemm block
    int i = 0;
    #pragma unroll 1
    do
    {
        LutEntry entry = Lut4s[i++];

        // Fetch 8 rows at a time from W and X
        TW w0;
        TX x0, x2, x4, x6;
        w0 = W0[entry.offsetW];
        if (bn)
        {
            x0 = X0[entry.offsetX];
            x2 = X2[entry.offsetX];
            x4 = X4[entry.offsetX];
            x6 = X6[entry.offsetX];
        }
        // Convert to float if needed and store to shared.
        if (Fprop)
            storW2[0] = ew_mul(to_float(w0), entry.gate);
        else
        {
            // transpose the shared store of W
            float2 w2 = ew_mul(to_float(w0), entry.gate);
            storW1[0] = w2.x;
            storW1[8] = w2.y;
        }
        storX4[0*16] = to_float(x0);
        storX4[2*16] = to_float(x2);
        storX4[4*16] = to_float(x4);
        storX4[6*16] = to_float(x6);

        float regX[4];
        float regW[4];

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
    }  while (i < lut_size);

    asm volatile ("\nEND_LOOP:\n":: );
    asm volatile ("mov.u32 %0, %tid.x;"   : "=r"(tid  ) :);
    asm volatile ("mov.u32 %0, %ctaid.x;" : "=r"(idx_N) :);

    // tid   = threadIdx.x;
    // idx_N = blockIdx.x;
    tid15 = tid >> 1;
    tid16 = tid  & 1;
    N2    = N >> 1;

    int n = idx_N*32 + tid15;
    int yi[4];
    yi[0] = (idx_K*8 + tid16*2)*N2 + n;
    yi[1] = yi[0] + N2;
    yi[2] = yi[0] + N2*4;
    yi[3] = yi[2] + N2;

    //printf("K:%2d N:%d tid:%2d t15:%2d t16:%2d N2:%2d n:%2d yi:%d\n", idx_K, idx_N, tid, tid15, tid16, N2, n, yi[0]);

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
        __syncwarp();

        int offsetC = locks*gridDim.x;
        int* Count = Lock + offsetC;
        int  count = *Count;
        __syncwarp();

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
            __syncwarp();

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
            __syncwarp();

            if (tid == 0)
                atomicExch(Lock, 0);
            // End Critial Section
        }
    }
}



template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(32) gemm_blocksparse_gated_08x64x08x8_updat(
    struct Plist<TX,8> X, struct Plist<TE,8> E,
    const  int2* __restrict__ Lut,
    const float* __restrict__ Gate,
    TU* U,
    int params8, int N, int loops, float alpha, float beta)
{
    // add padding for bank-conflict-free stores
    //__shared__ float2 shrU2[64*8 + 4*8]; // add padding for bank-conflict-free stores

    asm(".shared .align 16 .b32 share[1088];" ::); // 1088 = (64*8 + 4*8)*2

    extern __shared__ float2 shrU2[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float gate    = Gate[bid];
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

    // exit kernel if gate is zero
    asm ("{\n\t"
        ".reg .pred p;                      \n\t"
        "setp.eq.ftz.f32 p, %0, 0f00000000; \n\t"
        "@p exit;                           \n\t"
        "}" :: "f"(gate));

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
    //alpha *= gate;
    u2.x = alpha*u2.x + beta*b2.x;
    u2.y = alpha*u2.y + beta*b2.y;
    store(U, u2, offset);
}

template <typename TX, typename TE, typename TU>
__global__ void __launch_bounds__(32) gemm_blocksparse_gated_08x64x08x4_updat(
    struct Plist<TX,8> X, struct Plist<TE,8> E,
    const  int2* __restrict__ Lut,
    const float* __restrict__ Gate,
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

    float gate    = Gate[bid];
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

    // exit kernel if gate is zero
    asm ("{\n\t"
        ".reg .pred p;                      \n\t"
        "setp.eq.ftz.f32 p, %0, 0f00000000; \n\t"
        "@p exit;                           \n\t"
        "}" :: "f"(gate));

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
    //alpha *= gate;
    u2.x = alpha*u2.x + beta*b2.x;
    u2.y = alpha*u2.y + beta*b2.y;
    store(U, u2, offset);
}



template <bool Fprop, CTYPE(T)>
cudaError_t BsmmGatedXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params)
{
    dim3 grid(CEIL_DIV(params->N, 64), params->segments, 1);

    // printf("grid: %d %d\n", grid.x, grid.y);

    const int2* L2 = (const int2*)params->Lut;
    const   T2* W2 = (const   T2*)W;
    const   T4* X4 = (const   T4*)X;
    const   T8* X8 = (const   T8*)X;
            T2* Y2 = (        T2*)Y;
            T8* Y8 = (        T8*)Y;

    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, grid.x * params->locks * 2, params->stream);

    if (params->bsize == 8)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_gated_08x64x08x8_xprop<Fprop,T2,T8,T8><<<grid,32,params->shared*2,params->stream>>>(L2, params->Gate, W2, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_gated_08x64x08x4_xprop<Fprop,T2,T4,T2><<<grid,32,params->shared*2,params->stream>>>(L2, params->Gate, W2, X4, Y2, params->Lock, params->locks, params->N);
    }
    return cudaPeekAtLastError();
}
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);

template cudaError_t BsmmGatedXprop_CN<false, VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<false, VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<false, VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);


template <CTYPE(T)>
cudaError_t BsmmGatedUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params)
{
    dim3 grid(params->blocks, 1, 1);
    int loops = CEIL_DIV(params->N, 64);

    struct Plist<T4,8>* X4 = (struct Plist<T4,8>*)X;
    struct Plist<T4,8>* E4 = (struct Plist<T4,8>*)E;
    struct Plist<T8,8>* X8 = (struct Plist<T8,8>*)X;
    struct Plist<T8,8>* E8 = (struct Plist<T8,8>*)E;

    const int2* L2 = (const int2*)params->Lut;
            T2* U2 = (        T2*)U;

    if (params->bsize == 8)
    {
        // If not accumulating zero out the buffer
        if (params->beta == 0.0f)
            cuMemsetD8Async((CUdeviceptr)U, 0, params->blocks * 64 * sizeof(T), params->stream);

        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_gated_08x64x08x8_updat<T8,T8,T2><<<grid,32,0,params->stream>>>(*X8, *E8, L2, params->Gate, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_gated_08x64x08x4_updat<T4,T4,T2><<<grid,32,0,params->stream>>>(*X4, *E4, L2, params->Gate, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    return cudaPeekAtLastError();
}
template cudaError_t BsmmGatedUpdat_CN<VTYPE(float)>(const float* X, const float* E, float* U, bsmm_params* params);
template cudaError_t BsmmGatedUpdat_CN<VTYPE(ehalf)>(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
template cudaError_t BsmmGatedUpdat_CN<VTYPE(bhalf)>(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);


#endif // GOOGLE_CUDA

