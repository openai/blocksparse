
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <type_traits>
//#include "tensorflow/core/lib/core/status.h"


template <bool Fprop, CTYPE3(TX, TW, TY)>
cudaError_t BsmmXprop_CN(const TX* X, const TW* W, TY* Y, bsmm_params* params);

template <CTYPE3(TX, TE, TU)>
cudaError_t BsmmUpdat_CN(const TX* X, const TE* E, TU* U, bsmm_params* params);

template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmul
{
public:
    BlocksparseMatmul(bsmm_params* params) : params_(params) {}
    virtual ~BlocksparseMatmul() {}

    virtual Status Compute(const TA* A, const TB* B, TC* C) =0;

    bsmm_params* params_;
};


template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulFprop_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulFprop_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        cudaError_t res = BsmmXprop_CN<true,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulBprop_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulBprop_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        cudaError_t res = BsmmXprop_CN<false,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulUpdat_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulUpdat_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        cudaError_t res = BsmmUpdat_CN<VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};





Status GetKernel(std::string& kernel_name, CUfunction* kernel);

template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmul_NC : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmul_NC(bsmm_params* params, const char* op, int depth, int threads) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params), threads_(threads)
    {
        const char* dtypeA = std::is_same<TA, ehalf>::value ? "A10" : std::is_same<TA, bhalf>::value ? "A7": "A32";
        const char* dtypeB = std::is_same<TB, ehalf>::value ? "B10" : std::is_same<TB, bhalf>::value ? "B7": "B32";
        const char* dtypeC = std::is_same<TC, ehalf>::value ? "C10" : std::is_same<TC, bhalf>::value ? "C7": "C32";

        // int depth;
        // const char* op;
        // if      (mode_  == 0) { op = "fprop"; depth = 32; threads_ = 128; }
        // else if (mode_  == 1) { op = "bprop"; depth = 32; threads_ = 128; }
        // else                  { op = "updat"; depth =  8; threads_ =  32; }

        char kernel_name[48];
        sprintf(kernel_name, "gemm_blocksparse_32x32x%d_%s_%s_%s_%s", depth, op, dtypeA, dtypeB, dtypeC);
        kernel_name_ = kernel_name;
        kernel_ = 0;
    }
    Status Xprop_Kernel(const TA* A, const TB* B, TC* C)
    {
        GetKernel(kernel_name_, &kernel_);
        //printf("%s %p\n", kernel_name_.c_str(), kernel_);

        bsmm_params* params = this->params_;

        int gridX = (params->N >> 5) + ((params->N & 31) != 0);
        int gridY = (params->K >> 5);

        void *args[] = { &params->Lut, &C, &A, &B, &params->alpha, &params->beta, &params->C, &params->K, &params->N };

        CUresult res = cuLaunchKernel(kernel_, gridX, gridY, 1, threads_, 1, 1, params->shared, params->stream, args, NULL);
        if (res != CUDA_SUCCESS)
        {
            const char* errstr;
            cuGetErrorString(res, &errstr);
            return errors::Internal(errstr);
        }
        return Status::OK();
    }
    virtual Status Compute(const TA* A, const TB* B, TC* C) =0;

    int threads_, gridX_, gridY_;
    std::string kernel_name_;
    CUfunction kernel_;
};

template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulFprop_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulFprop_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "fprop", 32, 128) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        return this->Xprop_Kernel(A, B, C);
    }
};
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulBprop_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulBprop_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "bprop", 32, 128) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        return this->Xprop_Kernel(A, B, C);
    }
};
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulUpdat_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulUpdat_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "updat", 8, 32) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        struct plist8<TA>* pA = (struct plist8<TA>*)A;
        struct plist8<TB>* pB = (struct plist8<TB>*)B;
        bsmm_params* params = this->params_;
        int pcount = params->pcount * 8;

        //printf("%p %p %p %p %d %d\n", pA->a[0], pB->a[0], L, C, N, params);

        GetKernel(this->kernel_name_, &this->kernel_);
        //printf("%s %p\n", kernel_name_.c_str(), kernel_);

        void *args[] = { pA, pB, &params->Lut, &C, &params->alpha, &params->beta, &params->C, &params->K, &params->N, &pcount };

        CUresult res = cuLaunchKernel(this->kernel_, params->blocks, 1, 1, this->threads_, 1, 1, params->shared, params->stream, args, NULL);
        if (res != CUDA_SUCCESS)
        {
            const char* errstr;
            cuGetErrorString(res, &errstr);
            return errors::Internal(errstr);
        }
        return Status::OK();
    }
};

#if DINTEL_MKL

#include "mkl_cblas.h"

typedef struct LutHead
{
    int lut_offset;
    int lut_size;
    int idx_Y;
    int idx_Lock;
} LutHead;

typedef struct LutEntry
{
    int idx_X;
    int idx_W;
} LutEntry;

template <int BLOCK_SIZE, CTYPE3(TA,TB,TC)>
class BlocksparseMatmulFprop_CPU : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulFprop_CPU(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        bsmm_params* params = this->params_;

        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, params->K, params->C, 1.0f, A, params->C, B, params->K, 0.0f, C, params->K);
        // return Status::OK();

        // cblas_sgemv(CblasRowMajor, CblasTrans, params->C, params->K, 1.0f, B, params->K, A, 1, 0.0f, C, 1);
        // return Status::OK();

        // Just support gemv for the moment... easy to switch this to gemm later.
        if (params->N > 1)
            return errors::Internal("Only minibatch size 1 supported on CPU");

        const int* Lut      = params->Lut;
        const int  segments = params->segments;

        int last_Y = -1;
        float* Y = NULL;
        for (int segment = 0; segment < segments; segment++)
        {
            LutHead lut_head = ((const LutHead*)Lut)[segment];
            const LutEntry* lut_entry = (const LutEntry*)Lut + lut_head.lut_offset;

            float beta = 1.0f;
            if (lut_head.idx_Y != last_Y)
            {
                last_Y = lut_head.idx_Y;
                Y = C + lut_head.idx_Y * BLOCK_SIZE;
                beta = 0.0f;
            }

            for (int i = 0; i < lut_head.lut_size; i++)
            {
                LutEntry entry = lut_entry[i];

                const TA* X = A + entry.idx_X * BLOCK_SIZE;
                const TB* W = B + entry.idx_W * BLOCK_SIZE * BLOCK_SIZE;

                cblas_sgemv(CblasRowMajor, CblasTrans, BLOCK_SIZE, BLOCK_SIZE, 1.0f, W, BLOCK_SIZE, X, 1, beta, Y, 1);
                beta = 1.0f;
            }
        }
        return Status::OK();
    }
};

#endif // DINTEL_MKL

// typedef union fp_bits {
//     float  f;
//     uint   i;
//     ushort s[2];
// } fp_bits;

// bhalf to_bhalf(float v)
// {
//     bhalf r;
//     fp_bits bits;
//     bits.f  = v;
//     bits.i &= 0xff800000; // sign + exponent
//     bits.f = v + bits.f * 0.00390625f; // round (0.00390625f = (127 - 7 - 1) << 23)
//     r.x = bits.s[1]; // truncate
//     return r;
// }
// float to_float(bhalf v)
// {
//     fp_bits bits;
//     bits.s[0] = 0;
//     bits.s[1] = v.x;
//     return bits.f;
// }

// bhalf2 to_bhalf(float2 v)
// {
//     bhalf2 r2;
//     fp_bits bitx, bity;
//     bitx.f  = v.x;
//     bity.f  = v.y;
//     bitx.i &= 0xff800000;
//     bity.i &= 0xff800000;
//     bitx.f = v.x + bitx.f * 0.00390625f;
//     bity.f = v.y + bity.f * 0.00390625f;
//     r2.x = (bity.i & 0xffff0000) | (bitx.i >> 16);
//     return r2;
// }
// float2 to_float(bhalf2 v)
// {
//     float2 r2;
//     fp_bits bitx, bity;
//     bitx.i = v.x << 16;
//     bity.i = v.x & 0xffff0000;
//     r2.x = bitx.f;
//     r2.y = bity.f;
//     return r2;
// }

// float load(const float* x, int i) { return x[i]; }
// float load(const bhalf* x, int i) { return to_float(x[i]); }
// float load(const ehalf* x, int i) { return 0.0f; }

// void store(float* x, int i, float v) { x[i] = v; }
// void store(bhalf* x, int i, float v) { x[i] = to_bhalf(v); }
// void store(ehalf* x, int i, float v) { }


// template <int BLOCK_SIZE, CTYPE3(TA,TB,TC)>
// class BlocksparseMatmulFprop_CPU : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
// {
// public:
//     BlocksparseMatmulFprop_CPU(bsmm_params* params) :
//         BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

//     virtual Status Compute(const TA* A, const TB* B, TC* C)
//     {
//         bsmm_params* params = this->params_;

//         if (params->N > 1)
//             return errors::Internal("Only minibatch size 1 supported on CPU");

//         const int* Lut      = params->Lut;
//         const int  segments = params->segments;

//         int segcount = 0;
//         int last_Y = -1;
//         for (int segment = 0; segment < segments; segment++)
//         {
//             LutHead lut_head = ((const LutHead*)Lut)[segment];
//             const LutEntry* lut_entry = (const LutEntry*)Lut + lut_head.lut_offset;

//             if (lut_head.idx_Y != last_Y)
//             {
//                 last_Y = lut_head.idx_Y;
//                 segcount = 0;
//             }

//             TC* Y = C + lut_head.idx_Y * BLOCK_SIZE;
//             float  y[BLOCK_SIZE] = {0};

//             for (int i = 0; i < lut_head.lut_size; i++)
//             {
//                 LutEntry entry = lut_entry[i];

//                 const TA* X = A + entry.idx_X * BLOCK_SIZE;
//                 const TB* W = B + entry.idx_W * BLOCK_SIZE * BLOCK_SIZE;

//                 for (int c = 0; c < BLOCK_SIZE; c++)
//                 {
//                     float x = load(X, c);
//                     for (int k = 0; k < BLOCK_SIZE; k++)
//                         y[k] += load(W, c*BLOCK_SIZE + k) * x;
//                 }
//             }
//             if (segcount == 0)
//             {
//                 for (int k = 0; k < BLOCK_SIZE; k++)
//                     store(Y, k, y[k]);
//             }
//             else
//             {
//                 for (int k = 0; k < BLOCK_SIZE; k++)
//                     store(Y, k, load(Y, k) + y[k]);
//             }
//             segcount++;
//         }
//         return Status::OK();
//     }
// };