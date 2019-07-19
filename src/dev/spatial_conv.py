#!/usr/bin/env python

# Experimental depthwise seperable convolution kernels (just the spatial components) that run on tensorcores.
# (C,H,W,N) format is used, but if remapped to (N, heads, H, W, head_state) can be resused in self attention style convolution.
# Though the filters can no longer be broadcast, and relative attention will need to be added, just minor changes.
# It should also be possible to fuse in softmax.
# Stand-Alone Self-Attention in Vision Models: https://arxiv.org/abs/1906.05909

import os
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.autoinit import context, device

file_path = os.path.dirname(os.path.realpath(__file__))
attributes = drv.Context.get_device().get_attributes()
SMs = attributes[drv.device_attribute.MULTIPROCESSOR_COUNT]

print(device.name(), SMs)

# for name in sorted(list(attributes.keys())):
#     print(name, attributes[name])
# exit()

def ceil_div(x, y):
    return -(-x // y)

def out_dim(S, W, padding, stride=1):
    return ceil_div(W - S + 1 + 2*padding, stride)

def fprop_slice(q, X, S, padding, stride):
    qs = q * stride - padding
    x1 = None
    for s in range(S):
        x = qs + s
        if x1 is None and x >= 0:
            x1 = x
            f1 = s
        if x < X:
            x2 = x
            f2 = s
    return (slice(f1, f2 + 1), slice(x1, x2 + 1), f2 - f1 + 1)

def bprop_slice(x, Q, S, padding, stride):
    #pad_eff = S - padding - 1
    xs = x - padding
    f, e = list(), list()
    for s in range(S):
        q = xs + s
        if q % stride == 0:
            q //= stride
            if q >= 0 and q < Q:
                f.append(s)
                e.append(q)
    if len(f) == 0:
        return (slice(0, 0, 1), slice(0, 0, 1), None)
    if len(f) == 1:
        fstride = estride = 1
    else:
        fstride = f[1] - f[0]
        estride = e[1] - e[0]
    return (slice(f[0], f[-1]+1, fstride), slice(e[0], e[-1]+1, estride), None)


def conv_spatial_xprop(buffers, params, fprop=True):

    C, H, W, N, P, Q, R, S, pad_r, pad_s, std = params
    F, I = buffers
    O = np.empty((C, P, Q, N), dtype=np.float32)

    if fprop:
        xprop_slice = fprop_slice
    else:
        xprop_slice = bprop_slice
        F = F[:,::-1,::-1].copy() # invert RS

    pSlice = [ xprop_slice(p, H, R, pad_r, std) for p in range(P) ]
    qSlice = [ xprop_slice(q, W, S, pad_s, std) for q in range(Q) ]

    for c, p, q in np.ndindex(C, P, Q):
        sliceR, sliceH, _ = pSlice[p]
        sliceS, sliceW, _ = qSlice[q]

        slicedF = F[c,sliceR,sliceS].reshape(-1)
        slicedI = I[c,sliceH,sliceW].reshape(-1, N)
        O[c,p,q,:] = np.dot(slicedF, slicedI)

    return O

def conv_spatial_updat(buffers, params):

    C, H, W, N, P, Q, R, S, pad_r, pad_s, std = params
    E, I = buffers
    U = np.zeros((C, R, S), dtype=np.float32)

    pSlice = [ fprop_slice(p, H, R, pad_r, std) for p in range(P) ]
    qSlice = [ fprop_slice(q, W, S, pad_s, std) for q in range(Q) ]

    for c, p, q in np.ndindex(C, P, Q):
        sliceR, sliceH, rlen = pSlice[p]
        sliceS, sliceW, slen = qSlice[q]

        slicedI = I[c,sliceH,sliceW,:].reshape(-1, N)
        slicedE = E[c,p,q,:]
        U[c,sliceR,sliceS] += np.dot(slicedI, slicedE).reshape(rlen, slen)

    return U

xprop_config = {
     3 : (8, 2, 1, 16),
     4 : (8, 2, 1, 15),
     5 : (8, 2, 1, 12),
     6 : (8, 2, 1, 12),
     7 : (8, 2, 1,  8),
     8 : (8, 2, 1,  8),
     9 : (8, 2, 1,  6),
    10 : (4, 1, 2,  5),
    11 : (4, 1, 2,  2),
    12 : (4, 1, 3,  5),
    13 : (4, 1, 3,  3),
    14 : (8, 2, 1,  5),
    15 : (8, 2, 1,  4),
    16 : (8, 2, 1,  4),
    17 : (8, 2, 1,  2),
    18 : (4, 1, 4,  3),
    19 : (4, 1, 4,  2),
    20 : (4, 1, 4,  2),
    21 : (4, 1, 4,  2),
    22 : (8, 2, 3,  2),
    23 : (8, 2, 2,  2),
    24 : (8, 2, 2,  2),
    25 : (8, 2, 2,  2),
    26 : (4, 2, 2,  2),
    27 : (4, 2, 2,  1),
    28 : (4, 2, 2,  1),
}

def xprop_kernel(buffers, params, fprop=True, config=None):

    devF, devI, devO = buffers
    C, H, W, N, P, Q, R, S, pad_r, pad_s, std = params

    k_dim, warp_y, warp_x, blocks = xprop_config[R] if config is None else config

    cpuO  = np.empty((C, P, Q, N), dtype=np.float16)
    blk_p = ceil_div(P, 8 // (std if fprop else 1))
    blk_q = ceil_div(Q, 8 // (std if fprop else 1))
    blk_n = ceil_div(N, 16)
    N16   = 1 if N % 16 == 0 else 0

    parms = (devF, devI, devO, C, H, W, N, P, Q, W*N, Q*N, H*W*N, P*Q*N, pad_r, pad_s, blk_q)
    block = (warp_y*warp_x*32, 1, 1)
    grid  = (blk_p*blk_q, blk_n, C)
    name  = "conv_spatial_chwn_p8q8n16_k%d_xprop" % k_dim
    sig   = "PPPIIIIIIIIIIIII"

    kwargs = dict(R=R, S=S, STRIDE=std, N16=N16, WARP_Y=warp_y, WARP_X=warp_x, BLOCKS=blocks, FPROP=int(fprop))

    if k_dim == 8:
        code = "".join("#define %s %d\n" % i for i in kwargs.items()) + r"""

#include "../ew_op_gpu.h"
#include "../gpu_hmma.h"
#define MAX(a,b) ( ( (a) > (b) ) ? (a) : (b) )

extern "C"
__global__ void __launch_bounds__(WARP_X*WARP_Y*32, BLOCKS) conv_spatial_chwn_p8q8n16_k8_xprop(
    const float* __restrict__ F,
    const ehalf* __restrict__ I,
          ehalf*              O,
    int C, int H, int W, int N, int P, int Q, int WN, int QN, int HWN, int PQN, int pad_r, int pad_s, int blk_q)
{
    const int TILE_P = 8;
    const int TILE_Q = 8;
    const int TILE_N = 16;
    const int THREADS = WARP_X * WARP_Y * 32;

    const int TILE_H   = TILE_P + R - 1;
    const int TILE_W1  = TILE_Q + S - 1;
    const int TILE_W   = CEIL_DIV(TILE_W1, 8)*8;
    const int TILE_PQ  = TILE_P * TILE_Q;
    const int TILE_HW  = TILE_H * TILE_W;
    const int TILE_RW  =      R * TILE_W;
    const int TILE_X   = CEIL_DIV(TILE_RW, WARP_X*8)*8;
    const int TILE_Y   = TILE_P / WARP_Y;
    const int TILE_RW4 = TILE_X * WARP_X;

    const int STRD_RW  = TILE_RW4 & (16|8|4) ? TILE_RW4 : TILE_RW4 + 4;
    const int STRD_HW  = (TILE_HW + TILE_RW4 - TILE_RW) | 4;

    const int SIZE_F = STRD_RW*4 + 8;    // pad 4 on either end to account for the shifted filer copies
    const int SIZE_I = STRD_HW * TILE_N;
    const int SIZE_O = WARP_X == 1 ? 0 : WARP_X*TILE_PQ*TILE_N;

    const int F_LOOPS = CEIL_DIV(STRD_RW, THREADS);
    const int I_LOOPS = CEIL_DIV(STRD_HW, THREADS);

    __shared__  ehalf hShare[MAX(SIZE_F + SIZE_I, SIZE_O*2)];
    float* fShare = (float*)&hShare[0];

    int tid = threadIdx.x;

    for (int i = 0; i < CEIL_DIV(SIZE_F, THREADS*4); i++)
        *(ehalf4*)&hShare[(i*THREADS + tid)*4] = ehalf4(0);

    int idx_c   = blockIdx.z;
    int offsetF = idx_c*R*S;
    float filter[F_LOOPS];
    for (int f = 0; f < F_LOOPS; f++)
    {
        int idx = f*THREADS + tid;
        int r = (uint)idx / TILE_W;
        int s = (uint)idx % TILE_W;
        bool f_in = s < S && r < R;
        if (!FPROP)
        {
            r = R - r - 1;
            s = S - s - 1;
        }
        const float* Fp = F + (offsetF + r*S + s);
        asm("mov.b64 %0, %0;" : "+l"(Fp) : );

        filter[f] = 0.0f;
        if (f_in)
            filter[f] = __ldg(Fp);
    }
    int idx_pq = blockIdx.x;
    int idx_n  = blockIdx.y;
    int idx_p  = (uint)idx_pq / blk_q;
    int idx_q  = (uint)idx_pq % blk_q;
    if (idx_p & 1)
        idx_q = blk_q - idx_q - 1;

    int p0 = idx_p*TILE_P;
    int q0 = idx_q*TILE_Q;
    int h0 = p0 - pad_r;
    int w0 = q0 - pad_s;
    int n0 = idx_n*TILE_N;
    int offsetI = idx_c*HWN + n0;

    asm("mov.b32 %0, %0;" : "+r"(p0) : );
    asm("mov.b32 %0, %0;" : "+r"(q0) : );

    ehalf4 image[I_LOOPS][4];
    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        int y = (uint)idx / TILE_W;
        int x = (uint)idx % TILE_W;
        int h = h0 + y;
        int w = w0 + x;

        for (int j = 0; j < 4; j++)
            ew_zero(image[i][j]);

        if (STRIDE == 1 || FPROP)
        {
            const ehalf4* pI  = (const ehalf4*)(I + (offsetI + h*WN + w*N));
            asm("mov.b64 %0, %0;" : "+l"(pI) : );

            if ((TILE_W1 == TILE_W || x < TILE_W1) &&
                (i+1 < I_LOOPS || y < TILE_H) &&
                h >= 0 && h < H && w >= 0 && w < W)
            {
                image[i][0] = __ldg(pI);
                for (int j = 1; j < 4; j++)
                    if (N16 || n0 + j*4 < N) image[i][j] = __ldg(pI + j);
            }
        }
        else
        {
            const ehalf4* pI  = (const ehalf4*)(I + (offsetI + h*WN/STRIDE + w*N/STRIDE));
            asm("mov.b64 %0, %0;" : "+l"(pI) : );

            if ((TILE_W1 == TILE_W || x < TILE_W1) &&
                (i+1 < I_LOOPS || y < TILE_H) &&
                h % STRIDE == 0 && w % STRIDE == 0 &&
                h >= 0 && h/STRIDE < H && w >= 0 && w/STRIDE < W)
            {
                image[i][0] = __ldg(pI);
                for (int j = 1; j < 4; j++)
                    if (N16 || n0 + j*4 < N) image[i][j] = __ldg(pI + j);
            }
        }
    }

    __syncthreads();
    for (int f = 0; f < F_LOOPS; f++)
    {
        int idx = f*THREADS + tid;
        ehalf h_filter = to_ehalf(filter[f]);

        if (f+1 < F_LOOPS || idx < TILE_RW)
            for (int i = 0; i < 4; i++)
                hShare[STRD_RW*i + idx + i + 4] = h_filter;
    }
    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        if  (i+1 < I_LOOPS || idx < STRD_HW)
            for (int j = 0; j < 4; j++)
                *(ehalf4*)&hShare[idx*4 + STRD_HW*j*4 + SIZE_F] = image[i][j];
    }
    __syncthreads();

    int tid16_4 = (tid & 16)/4;
    int warp    = tid / 32;
    int warp_x  = warp / WARP_Y;
    int warp_y  = warp % WARP_Y;

    int f_shr = warp_x*TILE_X + (tid & 3)*STRD_RW + (tid & 4) + 4 - tid16_4;
    int i_shr = (warp_x*TILE_X + warp_y*TILE_Y*TILE_W + (tid & 7))*4 + ((tid & 8) + tid16_4)*STRD_HW;
    asm("mov.b32 %0, %0;" : "+r"(f_shr) : );
    asm("mov.b32 %0, %0;" : "+r"(i_shr) : );

    float acc[TILE_Y][8] = {0};
    for (int x = 0; x < TILE_X; x += 8)
    {
        for (int y = 0; y < TILE_Y; y++)
        {
            ehalf4 f4 = *(ehalf4*)&hShare[f_shr + x];
            ehalf4 i4 = *(ehalf4*)&hShare[i_shr + (y*TILE_W + x)*4 + SIZE_F];

            mma_m8n8k4_nn(acc[y], f4, i4);
        }
    }

    tid     = threadIdx.x;
    idx_n   = blockIdx.y;
    idx_c   = blockIdx.z;
    warp    = tid / 32;
    warp_x  = warp / WARP_Y;
    warp_y  = warp % WARP_Y;
    bool t4 = (tid & 4) != 0;

    float sum[TILE_Y][4];
    for (int y = 0; y < TILE_Y; y++)
    {
        for (int i = 0; i < 4; i++)
        {
            float swap = t4 ? acc[y][i + 0] : acc[y][i + 4];
            sum[y][i]  = t4 ? acc[y][i + 4] : acc[y][i + 0];
            sum[y][i] += shfl_xor(swap, 4);
        }
    }

    if (WARP_X == 1)
    {
        int p = p0 + warp_y * TILE_Y;
        int q = q0 + (tid & 1) + (tid & 16)/4;
        int n = idx_n*TILE_N + (tid & (2|4|8));
        bool bn = N16 || n < N;

        int offsetO = idx_c*PQN + n;

        if (STRIDE > 1 && TILE_Y >= STRIDE && FPROP)
        {
            for (int y = 0; y < TILE_Y/STRIDE; y++)
                for (int x = 0; x < 4/STRIDE; x++)
                    if (q%STRIDE == 0 && p/STRIDE + y < P && q/STRIDE + x < Q && bn)
                        store_half2(O + (offsetO + (p/STRIDE + y)*QN + (q/STRIDE + x)*N), to_half2(&sum[y*STRIDE][x*2]));
        }
        else
        {
            for (int y = 0; y < TILE_Y; y++)
                for (int x = 0; x < 2; x++)
                    if (p + y < P && q + x*2 < Q && bn)
                        store_half2(O + (offsetO + (p + y)*QN + (q + x*2)*N), to_half2(&sum[y][x*2]));
        }
    }
    else
    {
        int ox = (tid & (2|4|8));
        int oy = warp_x*TILE_PQ + warp_y*TILE_Q*TILE_Y + (tid & 1) + (tid & 16)/4;
        int o_shr = oy*TILE_N + ox;

        __syncthreads();
        for (int y = 0; y < TILE_Y; y++)
            for (int j = 0; j < 2; j++)
                *(float2*)&fShare[o_shr + (y*TILE_Q + j*2)*TILE_N] = *(float2*)&sum[y][j*2];
        __syncthreads();

        int tx = tid % 4;
        int ty = tid / 4;
        int tn = tx  * 4;
        int n  = idx_n*TILE_N + tn;
        if (N16 || n < N)
        {
            int offsetO = idx_c*PQN + n;
            const int O_LINES = THREADS/4;
            const int O_LOOPS = CEIL_DIV(TILE_PQ, O_LINES);
            for (int i = 0; i < O_LOOPS; i++)
            {
                int idx = i*O_LINES + ty;
                int pi = (uint)idx / TILE_Q;
                int qi = (uint)idx % TILE_Q;
                int p = p0 + pi;
                int q = q0 + qi;

                if ((i+1 < O_LOOPS || idx < TILE_PQ) && p < P && q < Q)
                {
                    float4 out[WARP_X];
                    for (int x = 0; x < WARP_X; x++)
                        out[x] = *(float4*)&fShare[(x*TILE_PQ + pi*TILE_Q + qi)*TILE_N + tn];

                    for (int x = 1; x < WARP_X; x++)
                        out[0] = ew_add(out[0], out[x]);

                    store_half4(O + (offsetO + p*QN + q*N), to_half4(out[0]));
                }
            }
        }
    }
}
"""

    if k_dim == 4:
        code = "".join("#define %s %d\n" % i for i in kwargs.items()) + r"""

#include "../ew_op_gpu.h"
#include "../gpu_hmma.h"
#define MAX(a,b) ( ( (a) > (b) ) ? (a) : (b) )

extern "C"
__global__ void __launch_bounds__(WARP_X*WARP_Y*32, BLOCKS) conv_spatial_chwn_p8q8n16_k4_xprop(
    const float* __restrict__ F,
    const ehalf* __restrict__ I,
          ehalf*              O,
    int C, int H, int W, int N, int P, int Q, int WN, int QN, int HWN, int PQN, int pad_r, int pad_s, int blk_q)
{
    const int TILE_P = 8;
    const int TILE_Q = 8;
    const int TILE_N = 16;
    const int THREADS = WARP_X * WARP_Y * 32;

    const int TILE_H   = TILE_P + R - 1;
    const int TILE_W1  = TILE_Q + S - 1;
    const int TILE_W   = CEIL_DIV(TILE_W1, 4)*4;
    const int TILE_PQ  = TILE_P * TILE_Q;
    const int TILE_HW  = TILE_H * TILE_W;
    const int TILE_RW  =      R * TILE_W;
    const int TILE_X   = CEIL_DIV(TILE_RW, WARP_X*4)*4;
    const int TILE_Y   = 4 / WARP_Y;
    const int TILE_RW4 = TILE_X * WARP_X;

    const int STRD_RW  = TILE_RW4 | 4;
    const int STRD_HW  = (TILE_HW + TILE_RW4 - TILE_RW) | 4;

    const int SIZE_F = STRD_RW*4 + 8;    // pad 4 on either end to account for the shifted filer copies
    const int SIZE_I = STRD_HW * TILE_N;
    const int SIZE_O = WARP_X*TILE_PQ*TILE_N;

    const int F_LOOPS = CEIL_DIV(STRD_RW, THREADS);
    const int I_LOOPS = CEIL_DIV(STRD_HW, THREADS);

    __shared__  ehalf hShare[MAX(SIZE_F + SIZE_I, SIZE_O*2)];
    float* fShare = (float*)&hShare[0];

    int tid = threadIdx.x;

    for (int i = 0; i < CEIL_DIV(SIZE_F, THREADS*4); i++)
        *(ehalf4*)&hShare[(i*THREADS + tid)*4] = ehalf4(0);

    int idx_c   = blockIdx.z;
    int offsetF = idx_c*R*S;
    float filter[F_LOOPS];
    for (int f = 0; f < F_LOOPS; f++)
    {
        int idx = f*THREADS + tid;
        int r = (uint)idx / TILE_W;
        int s = (uint)idx % TILE_W;
        bool f_in = s < S && r < R;
        if (!FPROP)
        {
            r = R - r - 1;
            s = S - s - 1;
        }
        const float* Fp = F + (offsetF + r*S + s);
        asm("mov.b64 %0, %0;" : "+l"(Fp) : );

        filter[f] = 0.0f;
        if (f_in)
            filter[f] = __ldg(Fp);
    }
    int idx_pq = blockIdx.x;
    int idx_n  = blockIdx.y;
    int idx_p  = (uint)idx_pq / blk_q;
    int idx_q  = (uint)idx_pq % blk_q;
    if (idx_p & 1)
        idx_q = blk_q - idx_q - 1;

    int p0 = idx_p*TILE_P;
    int q0 = idx_q*TILE_Q;
    int h0 = p0 - pad_r;
    int w0 = q0 - pad_s;
    int n0 = idx_n*TILE_N;
    int offsetI = idx_c*HWN + n0;

    asm("mov.b32 %0, %0;" : "+r"(p0) : );
    asm("mov.b32 %0, %0;" : "+r"(q0) : );

    ehalf4 image[I_LOOPS][4];
    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        int y = (uint)idx / TILE_W;
        int x = (uint)idx % TILE_W;
        int h = h0 + y;
        int w = w0 + x;
        const ehalf4* pI  = (const ehalf4*)(I + (offsetI + h*WN + w*N));
        asm("mov.b64 %0, %0;" : "+l"(pI) : );

        for (int j = 0; j < 4; j++)
            ew_zero(image[i][j]);

        if ((TILE_W1 == TILE_W || x < TILE_W1) &&
            (i+1 < I_LOOPS || y < TILE_H) &&
            h >= 0 && h < H && w >= 0 && w < W)
        {
            image[i][0] = __ldg(pI);
            for (int j = 1; j < 4; j++)
                if (N16 || n0 + j*4 < N) image[i][j] = __ldg(pI + j);
        }
    }

    __syncthreads();
    for (int f = 0; f < F_LOOPS; f++)
    {
        int idx = f*THREADS + tid;
        ehalf h_filter = to_ehalf(filter[f]);

        if (f+1 < F_LOOPS || idx < TILE_RW)
            for (int i = 0; i < 4; i++)
                hShare[STRD_RW*i + idx + i + 4] = h_filter;
    }
    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        if  (i+1 < I_LOOPS || idx < STRD_HW)
            for (int j = 0; j < 4; j++)
                *(ehalf4*)&hShare[idx*4 + STRD_HW*j*4 + SIZE_F] = image[i][j];
    }
    __syncthreads();

    int tid3    = tid & 3;
    int tid16_4 = (tid & 16)/4;
    int warp    = tid / 32;
    int warp_x  = warp / WARP_Y;
    int warp_y  = warp % WARP_Y;

    int f_shr = warp_x*TILE_X + tid3*STRD_RW + 4 - tid16_4;
    int i_shr = (warp_x*TILE_X + warp_y*TILE_Y*TILE_W + tid16_4*TILE_W + tid3)*4 + (tid & (4|8))*STRD_HW;
    asm("mov.b32 %0, %0;" : "+r"(f_shr) : );
    asm("mov.b32 %0, %0;" : "+r"(i_shr) : );

    float acc[TILE_Y][8] = {0};
    for (int x = 0; x < TILE_X; x += 4)
    {
        for (int y = 0; y < TILE_Y; y++)
        {
            ehalf4 f4 = *(ehalf4*)&hShare[f_shr + x];
            ehalf4 i4 = *(ehalf4*)&hShare[i_shr + (y*TILE_W + x)*4 + SIZE_F];

            mma_m8n8k4_nn(acc[y], f4, i4);
        }
    }

    tid     = threadIdx.x;
    tid16_4 = (tid & 16)/4;
    warp    = tid / 32;
    warp_x  = warp / WARP_Y;
    warp_y  = warp % WARP_Y;

    int ox = (tid & (2|4|8));
    int oy = warp_x*TILE_PQ + warp_y*TILE_Q*TILE_Y + (tid & 1) + tid16_4;
    int o_shr = oy*TILE_N + ox;

    __syncthreads();
    for (int y = 0; y < TILE_Y; y++)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                *(float2*)&fShare[o_shr + ((y + i*4)*TILE_Q + j*2)*TILE_N] = *(float2*)&acc[y][i*4 + j*2];
    __syncthreads();

    idx_n = blockIdx.y;
    idx_c = blockIdx.z;

    int tx = tid % 4;
    int ty = tid / 4;
    int tn = tx  * 4;
    int n  = idx_n*TILE_N + tn;
    if (N16 || n < N)
    {
        int offsetO = idx_c*PQN + n;
        const int O_LINES = THREADS/4;
        const int O_LOOPS = CEIL_DIV(TILE_PQ, O_LINES);
        for (int i = 0; i < O_LOOPS; i++)
        {
            int idx = i*O_LINES + ty;
            int pi = (uint)idx / TILE_Q;
            int qi = (uint)idx % TILE_Q;
            int p = p0 + pi;
            int q = q0 + qi;

            if ((i+1 < O_LOOPS || idx < TILE_PQ) && p < P && q < Q)
            {
                float4 out[WARP_X];
                for (int x = 0; x < WARP_X; x++)
                    out[x] = *(float4*)&fShare[(x*TILE_PQ + pi*TILE_Q + qi)*TILE_N + tn];

                for (int x = 1; x < WARP_X; x++)
                    out[0] = ew_add(out[0], out[x]);

                store_half4(O + (offsetO + p*QN + q*N), to_half4(out[0]));
            }
        }
    }
}
"""

    module = SourceModule(code, options=["--use_fast_math"], include_dirs=[file_path], no_extern_c=True, arch="compute_70", code="sm_70")
    kernel = module.get_function(name)
    kernel.prepare(sig)

    def func():
        kernel.prepared_call(grid, block, *parms)

    return func, devO, cpuO


updat_config = {
     3 : (4, 1,  64, 16),
     4 : (4, 2,  64, 16),
     5 : (4, 2,  64, 14),
     6 : (4, 4,  64,  8),
     7 : (4, 4,  64,  4),
     8 : (4, 4,  64,  4),
     9 : (8, 2,  96,  8),
    10 : (8, 2,  96,  8),
    11 : (8, 4,  64,  6),
    12 : (8, 4,  64,  4),
    13 : (8, 2, 160,  6),
    14 : (8, 2, 160,  6),
    15 : (8, 4,  96,  4),
    16 : (8, 4,  96,  3),
    17 : (8, 4, 128,  1),
    18 : (8, 4, 128,  2),
    19 : (8, 4, 128,  3),
    20 : (8, 4, 160,  1),
    21 : (8, 4, 160,  2),
    22 : (8, 4, 160,  3),
    23 : (8, 4, 192,  1),
    24 : (8, 4, 192,  2),
    25 : (8, 4, 224,  1),
    26 : (8, 4, 224,  2),
    27 : (8, 4, 256,  1),
    28 : (8, 4, 256,  1),
    29 : (8, 4, 288,  1),
}

def updat_kernel(buffers, params, config=None):

    devE, devI, devU = buffers
    C, H, W, N, P, Q, R, S, pad_r, pad_s, std = params

    tile_y, tile_x, threads, blocks = updat_config[R] if config is None else config

    cpuU  = np.empty((C, R, S), dtype=np.float32)
    blk_p = ceil_div(P, 8 // std)
    blk_q = ceil_div(Q, 8 // std)
    blk_n = ceil_div(N, 16)
    N16   = 1 if N % 16 == 0 else 0

    parms = (devE, devI, devU, C, H, W, N, P, Q, W*N, Q*N, H*W*N, P*Q*N, pad_r, pad_s, blk_q)
    block = (threads, 1, 1)
    grid  = (blk_p*blk_q, blk_n, C)
    name  = "conv_spatial_chwn_p8q8n16_updat"
    sig   = "PPPIIIIIIIIIIIII"

    conf = dict(R=R, S=S, STRIDE=std, N16=N16, TILE_Y=tile_y, TILE_X=tile_x, THREADS=threads, BLOCKS=blocks)
    code = "".join("#define %s %d\n" % i for i in conf.items()) + r"""

#include "../ew_op_gpu.h"
#include "../gpu_hmma.h"

#define MAX(a,b) ( ( (a) > (b) ) ? (a) : (b) )

extern "C"
__global__ void __launch_bounds__(THREADS,BLOCKS) conv_spatial_chwn_p8q8n16_updat(
    const ehalf* __restrict__ E,
    const ehalf* __restrict__ I,
          float*              U,
    int C, int H, int W, int N, int P, int Q, int WN, int QN, int HWN, int PQN, int pad_r, int pad_s, int blk_q)
{
    const int TILE_P = 8;
    const int TILE_Q = 8;
    const int TILE_N = 16;

    const int TILE_H  = TILE_P + R - 1;
    const int TILE_W  = TILE_Q + S - 1;
    const int TILE_HW = TILE_H * TILE_W;
    const int TILE_RW = TILE_W * R;
    const int TILE_PQ = TILE_P * TILE_Q;

    const int WARP_X = CEIL_DIV(TILE_RW, TILE_X*32);
    const int WARP_Y = TILE_P / TILE_Y;
    const int STRD_U = (CEIL_DIV(TILE_RW, 16)|1)*16;
    const int SIZE_I = TILE_HW * TILE_N; // ehalfs
    const int SIZE_E = TILE_PQ * TILE_N; // ehalfs
    const int SIZE_U = TILE_Q  * WARP_Y * STRD_U; // floats

    const int I_EXT = TILE_X*32 - TILE_RW % (TILE_X*32);
    const int I_PAD = I_EXT > TILE_PQ ? (I_EXT - TILE_PQ)*TILE_N : 0;

    const int E_LINES  = THREADS/2;
    const int E_LOOPS  = CEIL_DIV(TILE_PQ, E_LINES);
    const int I_LOOPS  = CEIL_DIV(TILE_HW, THREADS);

    __shared__  ehalf hShare[MAX(SIZE_I + SIZE_E + I_PAD, SIZE_U*2)];
    float* fShare = (float*)&hShare[0];

    int tid    = threadIdx.x;
    int idx_pq = blockIdx.x;
    int idx_n  = blockIdx.y;
    int idx_c  = blockIdx.z;
    int idx_p  = (uint)idx_pq / blk_q;
    int idx_q  = (uint)idx_pq % blk_q;
    if (idx_p & 1)
        idx_q = blk_q - idx_q - 1;

    int p0 = idx_p*TILE_P;
    int q0 = idx_q*TILE_Q;
    int h0 = p0 - pad_r;
    int w0 = q0 - pad_s;
    int n  = idx_n*TILE_N;
    int offsetI = idx_c*HWN + n;

    ehalf8 image[I_LOOPS][2];
    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        int y = (uint)idx / TILE_W;
        int x = (uint)idx % TILE_W;
        int h = h0 + y;
        int w = w0 + x;
        const ehalf8* pI  = (const ehalf8*)(I + (offsetI + h*WN + w*N));
        asm("mov.b64 %0, %0;" : "+l"(pI) : );

        ew_zero(image[i][0]);
        ew_zero(image[i][1]);
        if ((i+1 < I_LOOPS || idx < TILE_HW) &&
            h >= 0 && h < H &&
            w >= 0 && w < W)
        {
            image[i][0] = __ldg(pI);
            if (N16 || n + 8 < N)
                image[i][1] = __ldg(pI + 1);
        }
    }
    int tx = tid % 2;
    int ty = tid / 2;
    int tn = tx  * 8;
    n += tn;
    int offsetE = idx_c*PQN + n;
    ehalf8 error[E_LOOPS];
    for (int i = 0; i < E_LOOPS; i++)
    {
        int idx = i*E_LINES + ty;
        int p = p0 + idx / TILE_Q;
        int q = q0 + idx % TILE_Q;
        ew_zero(error[i]);

        if (STRIDE == 1)
        {
            const ehalf8* pE = (const ehalf8*)(E + (offsetE + p*QN + q*N));
            asm("mov.b64 %0, %0;" : "+l"(pE) : );

            if ((i+1 < E_LOOPS || idx < TILE_PQ) &&
                p < P && q < Q &&
                (N16 || n < N))
                error[i] = __ldg(pE);
        }
        else
        {
            const ehalf8* pE = (const ehalf8*)(E + (offsetE + p*QN/STRIDE + q*N/STRIDE));
            asm("mov.b64 %0, %0;" : "+l"(pE) : );

            if ((i+1 < E_LOOPS || idx < TILE_PQ) &&
                p % STRIDE == 0 && q % STRIDE == 0 &&
                p/STRIDE < P && q/STRIDE < Q &&
                (N16 || n < N))
                error[i] = __ldg(pE);
        }
    }

    for (int i = 0; i < I_LOOPS; i++)
    {
        int idx = i*THREADS + tid;
        if  (i+1 < I_LOOPS || idx < TILE_HW)
        {
            *(ehalf8*)&hShare[idx*8 + TILE_HW*0] = image[i][0];
            *(ehalf8*)&hShare[idx*8 + TILE_HW*8] = image[i][1];
        }
    }
    for (int i = 0; i < E_LOOPS; i++)
    {
        int idx = i*E_LINES + ty;
        if  (i+1 < E_LOOPS || idx < TILE_PQ)
            *(ehalf8*)&hShare[idx*16 + tn + SIZE_I] = error[i];
    }
    __syncthreads();

    int warp   = tid / 32;
    int warp_t = tid % 32;
    int warp_y = warp / WARP_X;
    int warp_x = warp % WARP_X;

    int e_shr = (warp_y*TILE_Y*TILE_Q + (tid & 3) + (tid & 16)/4)*16;
    int i_shr = (warp_y*TILE_Y*TILE_W + warp_x*TILE_X*32 + warp_t)*8;
    asm("mov.b32 %0, %0;" : "+r"(e_shr) : );
    asm("mov.b32 %0, %0;" : "+r"(i_shr) : );

    float acc[TILE_X][8] = {0};
    for (int n = 0; n < 2; n++)
    {
        for (int y = 0; y < TILE_Y; y++)
        {
            for (int x = 0; x < TILE_X; x++)
            {
                ehalf8 e8 = *(ehalf8*)&hShare[e_shr + (y*TILE_Q*16 + n*8 + SIZE_I)];
                ehalf8 i8 = *(ehalf8*)&hShare[i_shr + (y*TILE_W + x*32 + n*TILE_HW)*8];

                mma_m8n8k8_nt(acc[x], e8, i8);
            }
        }
    }

    int ux = warp_x*TILE_X*32 + (tid & (2|4|8));
    int uy = warp_y*TILE_Q + (tid & 1) + (tid & 16)/4;

    int u_shr = uy*STRD_U + ux;

    __syncthreads();
    for (int x = 0; x < TILE_X; x++)
        for (int i = 0; i < 2; i++)
            if (ux + x*32 + i*16 < TILE_RW)
                for (int j = 0; j < 2; j++)
                    *(float2*)&fShare[u_shr + x*32 + i*16 + j*2*STRD_U] = *(float2*)&acc[x][i*4 + j*2];
    __syncthreads();

    int offsetF = idx_c*R*S;
    const int F_LOOPS = CEIL_DIV(TILE_RW, THREADS);
    for (uint f = 0; f < F_LOOPS; f++)
    {
        int idx = f*THREADS + tid;
        int r = (uint)idx / TILE_W;
        int s = (uint)idx % TILE_W;

        if (r < R && s < S)
        {
            float update = 0.0f;
            for (int y = 0; y < WARP_Y; y++)
                for (int q = 0; q < TILE_Q; q++)
                    update += fShare[(y*TILE_Q + q)*STRD_U + q + idx];

            atomicRed(U, update, offsetF + r*S + s);
        }
    }
}
    """

    module = SourceModule(code, options=["--use_fast_math"], include_dirs=[file_path], no_extern_c=True, arch="compute_70", code="sm_70")
    kernel = module.get_function(name)
    kernel.prepare(sig)

    def func():
        drv.memset_d8(devU, 0, cpuU.nbytes)
        kernel.prepared_call(grid, block, *parms)

    return func, devU, cpuU


class Conv(object):
    def __init__(self, C, R, S, H, W, N, pad="SAME", std=1, repeat=1, ones=0):

        if pad.upper() == "SAME":
            pad_r = max((R-1) // 2, 1)
            pad_s = max((S-1) // 2, 1)
            P = ceil_div(H, std)
            Q = ceil_div(W, std)
        # VALID
        else:
            pad_r, pad_s = 0, 0
            P = out_dim(R, H, pad_r, std)
            Q = out_dim(S, W, pad_s, std)

        self.dimF   = (C, R, S)
        self.dimI   = (C, H, W, N)
        self.dimO   = (C, P, Q, N)
        self.std    = std
        self.repeat = repeat
        self.ones   = ones
        self.nflops = C * P * Q * R * S * N * 2.0

        self.fprop_params = (C, H, W, N, P, Q, R, S,   pad_r,     pad_s,   std)
        self.bprop_params = (C, P, Q, N, H, W, R, S, R-pad_r-1, S-pad_s-1, std)

        self.param_str = f"C:{C:d} R:{R:2d} S:{S:2d} H:{H:d} W:{W:d} P:{P:d} Q:{Q:d} N:{N:d}"

    def init(self):

        if self.ones:
            F = np.ones(self.dimF, dtype=np.float32)
            I = np.ones(self.dimI, dtype=np.float32)
            E = np.ones(self.dimO, dtype=np.float32)
            #E[:,0:4,:,:] = 0
            # print(E[0,0,:,:])
        else:
            # F = np.random.uniform(-1.0, 1.0, self.dimF).astype(np.float16).astype(np.float32)
            # I = np.random.uniform(-1.0, 1.0, self.dimI).astype(np.float16).astype(np.float32)
            # E = np.random.uniform(-1.0, 1.0, self.dimO).astype(np.float16).astype(np.float32)
            F = np.random.normal(0.0, 1.0, self.dimF).astype(np.float16).astype(np.float32)
            I = np.random.normal(0.0, 1.0, self.dimI).astype(np.float16).astype(np.float32)
            E = np.random.normal(0.0, 1.0, self.dimO).astype(np.float16).astype(np.float32)

        devF = drv.mem_alloc(F.size*4)
        devU = drv.mem_alloc(F.size*4)

        devI = drv.mem_alloc(I.size*2)
        devB = drv.mem_alloc(I.size*2)

        devO = drv.mem_alloc(E.size*2)
        devE = drv.mem_alloc(E.size*2)

        drv.memcpy_htod(devF, F)
        drv.memcpy_htod(devI, I.astype(np.float16))
        drv.memcpy_htod(devE, E.astype(np.float16))

        self.gpu_fprop_params = (devF, devI, devO)
        self.gpu_bprop_params = (devF, devE, devB)
        self.gpu_updat_params = (devE, devI, devU)

        self.cpu_fprop_params = (F, I)
        self.cpu_bprop_params = (F, E)
        self.cpu_updat_params = (E, I)

        self.events = (drv.Event(), drv.Event())
        self.nbytes = F.size*4 + I.size*2 + E.size*2

    def cleanup(self):

        self.gpu_fprop_params = None
        self.gpu_bprop_params = None
        self.gpu_updat_params = None
        self.cpu_fprop_params = None
        self.cpu_bprop_params = None
        self.cpu_updat_params = None
        self.events = None

    def execute(self, func, devO, cpuO, op):

        # warmup
        for r in range(self.repeat - 1):
            func()

        start, end = self.events
        start.record()
        for r in range(self.repeat):
            func()
        end.record()
        end.synchronize()
        msecs = end.time_since(start) / self.repeat
        if self.repeat > 1:
            gflops = self.nflops / (msecs * 1e6)
            gbps   = self.nbytes / (msecs * 2**30 * 0.001)
            res = f"{op} GFlops:{gflops:5.0f} GBps:{gbps:4.0f} ms:{msecs:7.3f} ({self.param_str})"
            print(res, flush=True)
            return (gflops, res)

        drv.memcpy_dtoh(cpuO, devO)
        if cpuO.dtype != np.float32:
            cpuO = cpuO.astype(np.float32)
        return cpuO

    def cpu_fprop(self):
        return conv_spatial_xprop(self.cpu_fprop_params, self.fprop_params, fprop=1)

    def cpu_bprop(self):
        return conv_spatial_xprop(self.cpu_bprop_params, self.bprop_params, fprop=0)

    def cpu_updat(self):
        return conv_spatial_updat(self.cpu_updat_params, self.fprop_params)

    def gpu_fprop(self):
        kernel, devO, cpuO = xprop_kernel(self.gpu_fprop_params, self.fprop_params, fprop=1)
        return self.execute(kernel, devO, cpuO, "F")

    def gpu_bprop(self):
        kernel, devB, cpuB = xprop_kernel(self.gpu_bprop_params, self.bprop_params, fprop=0)
        return self.execute(kernel, devB, cpuB, "B")

    def gpu_updat(self):
        kernel, devU, cpuU = updat_kernel(self.gpu_updat_params, self.fprop_params)
        return self.execute(kernel, devU, cpuU, "U")

    def gpu_fprop_tune(self):

        C, H, W, N, P, Q, R, S, pad_r, pad_s, std = self.fprop_params
        results = list()
        TILE_P  = 8;
        TILE_Q  = 8;
        TILE_N  = 16;
        TILE_H  = TILE_P + R - 1;
        TILE_W1 = TILE_Q + S - 1;
        TILE_W  = ceil_div(TILE_W1, 8)*8;
        TILE_PQ = TILE_P * TILE_Q;
        TILE_HW = TILE_H * TILE_W;
        TILE_RW =      R * TILE_W;
        #wys = (4,2,1) if R <= 10 else (2,1)
        for WARP_Y in (2,):
            for WARP_X in range(1,5):
                THREADS = WARP_X * WARP_Y * 32;

                TILE_X   = ceil_div(TILE_RW, WARP_X*8)*8
                TILE_Y   = TILE_P // WARP_Y
                TILE_RW4 = TILE_X  * WARP_X

                STRD_RW  = TILE_RW4 if TILE_RW4 & (16|8|4) else TILE_RW4 + 4
                STRD_HW  = (TILE_HW + TILE_RW4 - TILE_RW) | 4

                SIZE_F = STRD_RW*4 + 8;
                SIZE_I = STRD_HW * TILE_N;
                SIZE_O = WARP_X*TILE_PQ*TILE_N if WARP_X > 1 else 0

                F_LOOPS = ceil_div(STRD_RW, THREADS)
                I_LOOPS = ceil_div(STRD_HW, THREADS)

                SHARE  = max((SIZE_F + SIZE_I)*2, SIZE_O*4)
                BLOCKS = min(49152*2 // SHARE, 1024 // THREADS)
                if 64 <= THREADS <= 1024 and SHARE <= 49152:
                    for b in range(1, BLOCKS+1):
                        op = f"k:8 t:{THREADS:4d} wy:{WARP_Y:d} wx:{WARP_X:d} b:{b:2d} ma:{TILE_X//2:4d} fl:{F_LOOPS:2d} il:{I_LOOPS:2d} sh:{SHARE:5d}"
                        #print(op)

                        kernel, devO, cpuO = xprop_kernel(self.gpu_fprop_params, self.fprop_params, fprop=1, config=(8,WARP_Y,WARP_X,b))
                        results.append(self.execute(kernel, devO, cpuO, op))

        TILE_W  = ceil_div(TILE_W1, 4)*4;
        TILE_PQ = TILE_P * TILE_Q;
        TILE_HW = TILE_H * TILE_W;
        TILE_RW =      R * TILE_W;
        #wys = (4,2,1) if R <= 10 else (2,1)
        for WARP_Y in (2,1):
            for WARP_X in range(1,9):
                THREADS = WARP_X * WARP_Y * 32;

                TILE_X   = ceil_div(TILE_RW, WARP_X*4)*4
                TILE_Y   = 4 // WARP_Y
                TILE_RW4 = TILE_X  * WARP_X

                STRD_RW  = TILE_RW4 | 4
                STRD_HW  = (TILE_HW + TILE_RW4 - TILE_RW) | 4

                SIZE_F = STRD_RW*4 + 8;
                SIZE_I = STRD_HW * TILE_N;
                SIZE_O = WARP_X*TILE_PQ*TILE_N

                F_LOOPS = ceil_div(STRD_RW, THREADS)
                I_LOOPS = ceil_div(STRD_HW, THREADS)

                SHARE  = max((SIZE_F + SIZE_I)*2, SIZE_O*4)
                BLOCKS = min(49152*2 // SHARE, 1024 // THREADS)
                if 64 <= THREADS <= 1024 and SHARE <= 49152:
                    for b in range(1, BLOCKS+1):
                        op = f"k:4 t:{THREADS:4d} wy:{WARP_Y:d} wx:{WARP_X:d} b:{b:2d} ma:{TILE_X:4d} fl:{F_LOOPS:2d} il:{I_LOOPS:2d} sh:{SHARE:5d}"
                        #print(op)

                        kernel, devO, cpuO = xprop_kernel(self.gpu_fprop_params, self.fprop_params, fprop=1, config=(4, WARP_Y, WARP_X, b))
                        results.append(self.execute(kernel, devO, cpuO, op))
        print(R, S)
        for g, s in sorted(results):
            print(s)
        print("", flush=True)

    def gpu_updat_tune(self):

        C, H, W, N, P, Q, R, S, pad_r, pad_s, std = self.fprop_params
        TILE_P = 8
        TILE_Q = 8
        TILE_N = 16
        TILE_PQ = 64
        TILE_HW = (7 + S)*(7 + S)
        TILE_RW = (7 + S)*S
        results = list()
        for TILE_Y in (4,8):
            for TILE_X in (1,2,4):
                WARPS_X = ceil_div(TILE_RW, TILE_X*32)
                WARPS_Y = TILE_P // TILE_Y
                THREADS = WARPS_X * WARPS_Y * 32;
                STRIDE_U = (ceil_div(TILE_RW, 16)|1)*16;
                I_SIZE = TILE_HW * TILE_N
                E_SIZE = TILE_P * TILE_Q * TILE_N
                U_SIZE = TILE_Q  * WARPS_Y * STRIDE_U
                I_EXT = TILE_X*32 - TILE_RW % (TILE_X*32)
                I_PAD = (I_EXT - TILE_PQ)*TILE_N if I_EXT > TILE_PQ else 0
                SHARE = max((I_SIZE+E_SIZE+I_PAD)*2, U_SIZE*4)
                BLOCKS = min(49152*2 // SHARE, 2048 // THREADS)
                if 64 <= THREADS <= 1024 and SHARE <= 49152 and I_EXT / TILE_HW < 0.3:
                    for b in range(1, BLOCKS+1):
                        op = f"t:{THREADS:4d} ty:{TILE_Y:d} tx:{TILE_X:d} b:{b:2d} sh:{SHARE:5d}"

                        kernel, devU, cpuU = updat_kernel(self.gpu_updat_params, self.fprop_params, config=(TILE_Y, TILE_X, THREADS, b))
                        results.append( self.execute(kernel, devU, cpuU, op) )
        print(R, S)
        for g, s in sorted(results):
            print(s)
        print("", flush=True)


fbu     = (1, 1, 1)
ones    = 0
out     = 0
repeat  = 200

for S in range(3,26):
    for conv in (
        Conv(C=SMs*2, R=S, S=S, H=64, W=64, N=16, pad="SAME", std=1, repeat=repeat, ones=ones),
        Conv(C=SMs*2, R=S, S=S, H=64, W=64, N=16, pad="SAME", std=2, repeat=repeat, ones=ones),
    ):
        conv.init()

        if fbu[0]: devO = conv.gpu_fprop()
        if fbu[1]: devB = conv.gpu_bprop()
        if fbu[2]: devU = conv.gpu_updat()

        if repeat == 1:
            tests = list()
            if fbu[0]: tests.append(("F", devO, conv.cpu_fprop()))
            if fbu[1]: tests.append(("B", devB, conv.cpu_bprop()))
            if fbu[2]: tests.append(("U", devU, conv.cpu_updat()))

            for op, devO, cpuO in tests:
                difO = np.abs(cpuO - devO)

                l2_err = np.sqrt(np.square(difO).sum()) / np.sqrt(np.square(cpuO).sum())

                print(f"{op} max_err: {difO.max():8.3f} max_val: {devO.max():8.3f} l2_err: {l2_err:7.5f} {conv.param_str}")

                if out:
                    fmt = "%2.0f" if ones else "%5.2f"
                    np.savetxt("out_dif.txt", difO.reshape(-1, cpuO.shape[-1]), fmt=fmt)
                    np.savetxt("out_cpu.txt", cpuO.reshape(-1, cpuO.shape[-1]), fmt=fmt)
                    np.savetxt("out_gpu.txt", devO.reshape(-1, cpuO.shape[-1]), fmt=fmt)
                    exit()

        conv.cleanup()


# nsight-cli --section MemoryWorkloadAnalysis_Tables --details-all python spatial_conv.py | grep shared

# print(c[0,:,:,0])
# print()
# print(g[0,:,:,0])

# F GFlops: 2575 GBps: 533 ms:  0.073 (C:160 R: 3 S: 3 H:64 W:64 P:64 Q:64 N:16)
# U GFlops: 2660 GBps: 550 ms:  0.071 (C:160 R: 3 S: 3 H:64 W:64 P:64 Q:64 N:16)
# F GFlops: 4565 GBps: 532 ms:  0.074 (C:160 R: 4 S: 4 H:64 W:64 P:64 Q:64 N:16)
# U GFlops: 4711 GBps: 549 ms:  0.071 (C:160 R: 4 S: 4 H:64 W:64 P:64 Q:64 N:16)
# F GFlops: 7105 GBps: 530 ms:  0.074 (C:160 R: 5 S: 5 H:64 W:64 P:64 Q:64 N:16)
# U GFlops: 7349 GBps: 548 ms:  0.071 (C:160 R: 5 S: 5 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:10113 GBps: 524 ms:  0.075 (C:160 R: 6 S: 6 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:10125 GBps: 524 ms:  0.075 (C:160 R: 6 S: 6 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:13627 GBps: 518 ms:  0.075 (C:160 R: 7 S: 7 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:12927 GBps: 492 ms:  0.079 (C:160 R: 7 S: 7 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:17435 GBps: 508 ms:  0.077 (C:160 R: 8 S: 8 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:16277 GBps: 474 ms:  0.082 (C:160 R: 8 S: 8 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:22004 GBps: 507 ms:  0.077 (C:160 R: 9 S: 9 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:18627 GBps: 429 ms:  0.091 (C:160 R: 9 S: 9 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:20335 GBps: 379 ms:  0.103 (C:160 R:10 S:10 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:19008 GBps: 355 ms:  0.110 (C:160 R:10 S:10 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:22466 GBps: 346 ms:  0.113 (C:160 R:11 S:11 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:20461 GBps: 316 ms:  0.124 (C:160 R:11 S:11 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:25035 GBps: 325 ms:  0.121 (C:160 R:12 S:12 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:23536 GBps: 305 ms:  0.128 (C:160 R:12 S:12 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:27661 GBps: 306 ms:  0.128 (C:160 R:13 S:13 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:21994 GBps: 243 ms:  0.161 (C:160 R:13 S:13 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:28962 GBps: 276 ms:  0.142 (C:160 R:14 S:14 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:24807 GBps: 236 ms:  0.166 (C:160 R:14 S:14 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:30022 GBps: 249 ms:  0.157 (C:160 R:15 S:15 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:26382 GBps: 219 ms:  0.179 (C:160 R:15 S:15 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:30238 GBps: 221 ms:  0.178 (C:160 R:16 S:16 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:29406 GBps: 215 ms:  0.183 (C:160 R:16 S:16 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:35480 GBps: 230 ms:  0.171 (C:160 R:17 S:17 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:30019 GBps: 194 ms:  0.202 (C:160 R:17 S:17 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:31339 GBps: 181 ms:  0.217 (C:160 R:18 S:18 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:30584 GBps: 177 ms:  0.222 (C:160 R:18 S:18 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:32796 GBps: 170 ms:  0.231 (C:160 R:19 S:19 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:32548 GBps: 169 ms:  0.233 (C:160 R:19 S:19 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:35651 GBps: 167 ms:  0.235 (C:160 R:20 S:20 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:29178 GBps: 137 ms:  0.287 (C:160 R:20 S:20 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:37231 GBps: 158 ms:  0.248 (C:160 R:21 S:21 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:31712 GBps: 135 ms:  0.292 (C:160 R:21 S:21 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:32057 GBps: 124 ms:  0.317 (C:160 R:22 S:22 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:33891 GBps: 131 ms:  0.299 (C:160 R:22 S:22 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:35301 GBps: 125 ms:  0.314 (C:160 R:23 S:23 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:32931 GBps: 117 ms:  0.337 (C:160 R:23 S:23 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:35695 GBps: 116 ms:  0.338 (C:160 R:24 S:24 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:32320 GBps: 105 ms:  0.374 (C:160 R:24 S:24 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:39745 GBps: 120 ms:  0.330 (C:160 R:25 S:25 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:33094 GBps: 100 ms:  0.396 (C:160 R:25 S:25 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:33897 GBps:  94 ms:  0.418 (C:160 R:26 S:26 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:33407 GBps:  93 ms:  0.424 (C:160 R:26 S:26 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:34470 GBps:  89 ms:  0.444 (C:160 R:27 S:27 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:32564 GBps:  84 ms:  0.469 (C:160 R:27 S:27 H:64 W:64 P:64 Q:64 N:16)
# F GFlops:35377 GBps:  85 ms:  0.465 (C:160 R:28 S:28 H:64 W:64 P:64 Q:64 N:16)
# U GFlops:34760 GBps:  84 ms:  0.473 (C:160 R:28 S:28 H:64 W:64 P:64 Q:64 N:16)
