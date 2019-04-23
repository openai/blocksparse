
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using perftools::gputools::cuda::CUDAStream;

#include "gpu_types.h"

static void ClosestDivisorTo4(uint val, bool isA, uint* div, uint* res)
{
         if ((val % 4) == 0) { *div = 4; *res = val / 4; }
    else if ((val % 3) == 0) { *div = 3; *res = val / 3; }
    else if ((val % 5) == 0) { *div = 5; *res = val / 5; }
    else if ((val % 2) == 0) { *div = 2; *res = val / 2; }
    else if ((val % 7) == 0) { *div = 7; *res = val / 7; }
    else if (isA) { *div = val; *res =   1; }
    else          { *div = 1;   *res = val; }
}

#define FPROP_OP 0
#define BPROP_OP 1
#define UPDAT_OP 2

#define OP_N 0
#define OP_T 1

template <bool Fprop, CTYPE(T)>
cudaError_t BsmmXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params);

template <CTYPE(T)>
cudaError_t BsmmUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params);

template <bool Fprop, CTYPE(T)>
cudaError_t BsmmGatedXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params);

template <CTYPE(T)>
cudaError_t BsmmGatedUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_xn_64_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_64_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_64_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nt_64_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_64_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_64_dds(const float* X, const float* E, float* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_xn_128_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_128_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_xn_128_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nt_128_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_128_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_nt_128_dds(const float* X, const float* E, float* U, bsmm_params* params);

cudaError_t hgemm_blocksparse_nx_dsd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_nx_dsd(const float* X, const float* W, float* Y, bsmm_params* params, uint op);
cudaError_t hgemm_blocksparse_tn_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_tn_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);
cudaError_t hgemm_blocksparse_tn_dds(const float* X, const float* E, float* U, bsmm_params* params);

template <uint OP, MTYPE(T)>
class BlocksparseMatmulOp : public OpKernel
{
public:
    explicit BlocksparseMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), major_(0), repeat_(1), flops_(0.0f)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("segments", &params_.segments));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("locks",    &params_.locks   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &params_.blocks  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",    &params_.bsize  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("C",        &params_.C       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("K",        &params_.K       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shared",   &params_.shared  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",    &params_.alpha   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta",     &params_.beta    ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gated_dw", &gated_dw_       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_));
        params_.pcount = 1;
        params_.blk_A  = 0;

        is_gpu_ = ctx->device_type() == DEVICE_GPU;

        //OP_REQUIRES(ctx, axis_ == 0, errors::InvalidArgument("Only feature axis=0 currently supported."));

        // TODO: pack larger values of K in gridZ
        OP_REQUIRES(ctx, params_.K < params_.bsize*65536, errors::InvalidArgument("K < bsize*65536"));
        OP_REQUIRES(ctx, params_.C < params_.bsize*65536, errors::InvalidArgument("C < bsize*65536"));

        if (bench_)
        {
            repeat_ = bench_;
            flops_  = (float)(params_.blocks * params_.bsize*params_.bsize);

            const char* op = OP == FPROP_OP ? "FPROP" : OP == BPROP_OP ? "BPROP" : "UPDAT";
            sprintf(bench_string_, "%s %02d-%d C:%05d K:%05d blks:%d", op, params_.bsize, axis_, params_.C, params_.K, params_.blocks);
        }
    }
    void Compute(OpKernelContext* ctx) override
    {
        if (major_ == 0)
        {
            SMs_ = GetCountSMsVersion(&major_, NULL);
            //OP_REQUIRES(ctx, major_ >= 7, errors::InvalidArgument("Tensorcore GPU required"));
        }
        if (OP == UPDAT_OP)
            OP_REQUIRES_OK(ctx, this->Compute_Updat(ctx));
        else
            OP_REQUIRES_OK(ctx, this->Compute_Xprop(ctx, OP));
    }
    Status Compute_Xprop(OpKernelContext* ctx, uint op)
    {
        const Tensor& A = ctx->input(0);
        const Tensor& B = ctx->input(1);
        const Tensor& L = ctx->input(2);

        OpInputList gate;
        ctx->input_list("gate", &gate);

        TensorShape shapeC;
        int N     = 1;
        int rankA = A.dims();
        for (int i = 0; i < rankA; i++)
            if (i != axis_)
            {
                shapeC.AddDim(A.dim_size(i));
                N *= A.dim_size(i);
            }
            else
                shapeC.AddDim(params_.K);

        bool tensorcores = major_ >= 7 && std::is_same<T1, ehalf>::value;

        int blkN = 128, gridN = CEIL_DIV(N, 128), modN128 = N & 127;
        if (!tensorcores || axis_ == 1 || (modN128 > 0 && modN128 <= 64) || gridN * params_.segments < SMs_*4)
        {
            blkN  = 64;
            gridN = CEIL_DIV(N, 64);
        }

        Tensor* C;
        Status s = ctx->allocate_output(0, shapeC, &C);
        if (!s.ok()) return s;

        Tensor* Lock;
        TensorShape shapeL;
        if (params_.locks > 0)
            shapeL.AddDim(gridN * params_.locks * 2);
        s = ctx->allocate_output(1, shapeL, &Lock);
        if (!s.ok()) return s;

        params_.Lock = params_.locks > 0 ? Lock->flat<int32>().data() : nullptr;
        params_.N    = N;
        params_.Lut  = (const int*)L.flat<int64>().data();
        params_.Gate = gate.size() > 0 ? gate[0].flat<float>().data() : NULL;

        if (params_.blk_A == 0)
        {
            ClosestDivisorTo4(params_.segments, true, &params_.blk_a, &params_.blk_A);
            ClosestDivisorTo4(gridN,           false, &params_.blk_b, &params_.blk_B);

            // printf("%d %d %d %d %d %d\n", params_.segments, gridN, params_.blk_a, params_.blk_b, params_.blk_A, params_.blk_B);
        }

        const T1* pA = (const T1*)A.flat<T>().data();
        const T1* pB = (const T1*)B.flat<T>().data();
              T1* pC = (      T1*)C->flat<T>().data();

        if (is_gpu_)
            params_.stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        Benchmark* bench = nullptr;
        if (bench_) bench = new Benchmark(params_.stream, bench_string_, 0, flops_ * params_.N * params_.pcount, repeat_, is_gpu_);

        cudaError_t res;
        for (int r = 0; r < repeat_; r++)
            if (tensorcores)
            {
                if (axis_ == 0)
                    if (blkN == 64)
                        res = hgemm_blocksparse_xn_64_sdd( pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                    else
                        res = hgemm_blocksparse_xn_128_sdd(pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                else
                    res = hgemm_blocksparse_nx_dsd(pA, pB, pC, &params_, op == FPROP_OP ? OP_N : OP_T);
            }
            else
            {
                if (params_.Gate == NULL && axis_ == 0)
                {
                    if (op == FPROP_OP)
                        res = BsmmXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                    else
                        res = BsmmXprop_CN<false,NTYPE(T)>(pA, pB, pC, &params_);
                }
                else
                {
                    // Cuda update for Volta broke these kernels.  Need to fix.
                    // Ideally merge gated and non-gated code like is done with hgemm kernels.
                    return errors::Internal("Gated blocksparse matmul currently only supported on fp16 tensorcores.");
                    // if (op == NN_OP)
                    //     res = BsmmGatedXprop_CN<false,NTYPE(T)>(pA, pB, pC, &params_);
                    // else
                    //     res = BsmmGatedXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                }
            }

        if (bench) delete bench;

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
    Status Compute_Updat(OpKernelContext* ctx)
    {
        OpInputList x, dy, gate;

        ctx->input_list(   "x", &x);
        ctx->input_list(  "dy", &dy);
        ctx->input_list("gate", &gate);

        params_.pcount = x.size();

        if (params_.pcount > 8)
            return errors::Internal("No more than 8 inputs allowed.");

        struct Plist<T1,8> X;
        struct Plist<T1,8> DY;
        for (int i = 0; i < params_.pcount; ++i)
        {
             X.a[i] = (const T1*) x[i].flat<T>().data();
            DY.a[i] = (const T1*)dy[i].flat<T>().data();
        }
        params_.N = 1;
        int rank = x[0].dims();
        for (int i = 0; i < rank; i++)
            if (i != axis_)
                params_.N *= x[0].dim_size(i);

        T1* DW;
        if (params_.beta == 0.0f)
        {
            // BlocksparseMatmulDW: [x], [dy], lut, [gate]
            if (ctx->num_inputs() != params_.pcount*2 + 1 + gate.size())
                return errors::Internal("with beta=0.0, use BlocksparseMatmulDW ", ctx->num_inputs());

            Tensor* C;
            TensorShape shapeC({ params_.blocks, params_.bsize, params_.bsize });
            Status s = ctx->allocate_output(0, shapeC, &C);
            if (!s.ok()) return s;
            DW = (T1*)C->flat<T>().data();
        }
        else
        {
            // BlocksparseMatmulDWA: [x], [dy], lut, dwi, [gate]
            if (ctx->num_inputs() != params_.pcount*2 + 2 + gate.size())
                return errors::Internal("with beta!=0.0, use BlocksparseMatmulDWA ", ctx->num_inputs());

            // accumulate to C in place
            const Tensor& C = ctx->input(params_.pcount*2 + 1);
            ctx->set_output(0, C);
            DW = (T1*)C.flat<T>().data();
        }
        params_.Lut  = (const int*)ctx->input(params_.pcount*2).flat<int64>().data();
        params_.Gate = gated_dw_ && gate.size() > 0 ? gate[0].flat<float>().data() : NULL;

        if (is_gpu_)
            params_.stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        Benchmark* bench = nullptr;
        if (bench_) bench = new Benchmark(params_.stream, bench_string_, 0, flops_ * params_.N * params_.pcount, repeat_, is_gpu_);

        cudaError_t res;
        for (int r = 0; r < repeat_; r++)
            if (major_ >= 7 && std::is_same<T1, ehalf>::value)
            {
                if (axis_ == 0)
                {
                    int modN128 = params_.N & 127;
                    if (modN128 > 0 && modN128 <= 64)
                        res = hgemm_blocksparse_nt_64_dds( (const T1*)&X, (const T1*)&DY, DW, &params_);
                    else
                        res = hgemm_blocksparse_nt_128_dds((const T1*)&X, (const T1*)&DY, DW, &params_);
                }
                else
                    res = hgemm_blocksparse_tn_dds((const T1*)&X, (const T1*)&DY, DW, &params_);
            }
            else
            {
                if (params_.Gate == NULL && axis_ == 0)
                    res = BsmmUpdat_CN<NTYPE(T)>((const T1*)&X, (const T1*)&DY, DW, &params_);
                else
                    return errors::Internal("Gated blocksparse matmul currently only supported on fp16 tensorcores.");
                    // res = BsmmGatedUpdat_CN<NTYPE(T)>((const T1*)&X, (const T1*)&DY, DW, &params_);
            }

        if (bench) delete bench;

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
    bsmm_params params_;
    int   axis_, bench_, repeat_, SMs_, major_, grid_n_;
    float flops_;
    bool  gated_dw_, is_gpu_;
    char  bench_string_[256];
};

Status XpropShape(InferenceContext* ctx)
{
    int    K; TF_RETURN_IF_ERROR(ctx->GetAttr(   "K",    &K));
    int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

    // C ==> K
    ShapeHandle x = ctx->input(0);
    int rank = ctx->Rank(x);
    //printf("XpropShape: %d\n", rank);
    if (rank > 0)
    {
        std::vector<DimensionHandle> shape;
        shape.reserve(rank);
        for (int i = 0; i < rank; i++)
            shape.push_back(i == axis ? ctx->MakeDim(K) : ctx->Dim(x, i));

        ctx->set_output(0, ctx->MakeShape(shape));
    }
    else
        ctx->set_output(0, ctx->UnknownShape());
    ctx->set_output(1, ctx->UnknownShape());
    return Status::OK();
}
Status UpdatShape(InferenceContext* ctx)
{
    //printf("UpdatShape: %d\n", ctx->Rank(ctx->input(0)));

    int blocks, bsize;
    TF_RETURN_IF_ERROR(ctx->GetAttr("blocks", &blocks));
    TF_RETURN_IF_ERROR(ctx->GetAttr("bsize",  &bsize));

    // (blocks, block_size, block_size)
    DimensionHandle bsize_dim = ctx->MakeDim(bsize);
    ctx->set_output(0, ctx->MakeShape({ ctx->MakeDim(blocks), bsize_dim, bsize_dim }));
    return Status::OK();
}

REGISTER_OP("BlocksparseMatmul")
    .Input("x: T")
    .Input("w: T")
    .Input("lut: int64")
    .Input("lut_dx: int64")
    .Input("lut_dw: int64")
    .Input("gate: ngate * float")
    .Output("y: T")
    .Output("temp: int32")
    .Attr("T: {half, float, bfloat16}")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("segments_dx: int = 0")
    .Attr("locks: int = 0")
    .Attr("locks_dx: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("shared_dx: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("gated_dw: bool = false")
    .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    .Attr("ngate: int >= 0")
    .SetShapeFn(XpropShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),BlocksparseMatmulOp<FPROP_OP, FLOAT_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BlocksparseMatmulOp<FPROP_OP, EHALF_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BlocksparseMatmulOp<FPROP_OP, BHALF_V>);


REGISTER_OP("BlocksparseMatmulDX")
    .Input("dy: T")
    .Input("w: T")
    .Input("lut: int64")
    .Input("gate: ngate * float")
    .Output("dx: T")
    .Output("temp: int32")
    .Attr("T: {half, float, bfloat16}")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("gated_dw: bool = false")
    .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    .Attr("ngate: int >= 0")
    .SetShapeFn(XpropShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),BlocksparseMatmulOp<BPROP_OP, FLOAT_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BlocksparseMatmulOp<BPROP_OP, EHALF_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BlocksparseMatmulOp<BPROP_OP, BHALF_V>);


REGISTER_OP("BlocksparseMatmulDW")
    .Input("x: params * T")
    .Input("dy: params * T")
    .Input("lut: int64")
    .Input("gate: ngate * float")
    .Output("dw: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("params: int")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("gated_dw: bool = false")
    .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    .Attr("ngate: int >= 0")
    .SetShapeFn(UpdatShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),BlocksparseMatmulOp<UPDAT_OP, FLOAT_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BlocksparseMatmulOp<UPDAT_OP, EHALF_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BlocksparseMatmulOp<UPDAT_OP, BHALF_V>);


REGISTER_OP("BlocksparseMatmulDWA")
    .Input("x: params * T")
    .Input("dy: params * T")
    .Input("lut: int64")
    .Input("dwi: T")  // dw input to accumulate on top of
    .Input("gate: ngate * float")
    .Output("dw: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("params: int")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 1.0")
    .Attr("gated_dw: bool = false")
    .Attr("gate_grad: bool = false")
    .Attr("bench: int = 0")
    .Attr("ngate: int >= 0")
    .SetShapeFn(UpdatShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b" and accumulate to "c".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),BlocksparseMatmulOp<UPDAT_OP, FLOAT_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BlocksparseMatmulOp<UPDAT_OP, EHALF_V>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BlocksparseMatmulOp<UPDAT_OP, BHALF_V>);


template <typename T> bool BlocksparseGateGrad(CUstream stream, T* dw_out, float* dg, const T* dw, const T* w, const float* g, uint blocks, uint bsize);

REGISTER_OP("BlocksparseMatmulDG")
    .Input("dw: T")
    .Input("w: T")
    .Input("g: float")
    .Output("dw_out: T")
    .Output("dg: float")
    .Attr("T: { float, half }")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Blocksparse Gate Grad
)doc");

template <typename T, typename V>
class BlocksparseMatmulDGOp : public OpKernel {
 public:
  explicit BlocksparseMatmulDGOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& dw = ctx->input(0);
    const Tensor&  w = ctx->input(1);
    const Tensor&  g = ctx->input(2);

    uint blocks = dw.dim_size(0);
    uint bsize  = dw.dim_size(1);

    Tensor *dw_out;
    Tensor *dg;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dw.shape(), &dw_out));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  g.shape(), &dg));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseGateGrad<V>(stream,
      (V*)dw_out->flat<T>().data(),
      dg->flat<float>().data(),
      (const V*)dw.flat<T>().data(),
      (const V*) w.flat<T>().data(),
      g.flat<float>().data(),
      blocks, bsize
    );
  }
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDG").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), BlocksparseMatmulDGOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDG").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), BlocksparseMatmulDGOp<EHALF,ehalf>);



bool IdentityInitCK(CUstream stream, float* W, const int* lut, int CB, int KB, int blocks, int bsize, float scale);

REGISTER_OP("BlocksparseMatmulIdentityInit")
    .Input("lut: int32")
    .Output("w: float")
    .Attr("CB: int >=0")
    .Attr("KB: int >=0")
    .Attr("blocks: int >=0")
    .Attr("bsize: int")
    .Attr("scale: float = 1.0")
    .SetShapeFn(UpdatShape)
    .Doc(R"doc(
Identity Init a blocksparse weight matrix.
)doc");

class BlocksparseMatmulIdentityInitOp : public OpKernel {
 public:
  explicit BlocksparseMatmulIdentityInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("CB",     &CB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("KB",     &KB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks", &blocks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale",  &scale_ ));
  }

  void Compute(OpKernelContext* ctx) override {

    TensorShape c_shape({ blocks_, bsize_, bsize_ });

    Tensor* w = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &w));

        float*   w_ptr = w->flat<float>().data();
    const int* lut_ptr = ctx->input(0).flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    IdentityInitCK(stream, w_ptr, lut_ptr, CB_, KB_, blocks_, bsize_, scale_);
  }
 private:

  int blocks_, bsize_, CB_, KB_;
  float scale_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulIdentityInit").Device(DEVICE_GPU), BlocksparseMatmulIdentityInitOp);

Status ReducedDWShape(InferenceContext* ctx)
{
    int params, bsize, axis;
    TF_RETURN_IF_ERROR(ctx->GetAttr("n_params", &params));
    TF_RETURN_IF_ERROR(ctx->GetAttr("bsize",    &bsize));
    TF_RETURN_IF_ERROR(ctx->GetAttr("axis",     &axis));
    int bshift = bsize == 8 ? 3 : bsize == 16 ? 4 : bsize == 32 ? 5 : 6;

    ShapeHandle x = ctx->input(0);
    ShapeHandle y = ctx->input(params);
    int rank = ctx->Rank(x);
    if (rank > 1)
    {
        DimensionHandle C = ctx->MakeDim(ctx->Value(ctx->Dim(x, axis)) >> bshift);
        DimensionHandle K = ctx->MakeDim(ctx->Value(ctx->Dim(y, axis)) >> bshift);
        DimensionHandle P = ctx->MakeDim(params);

        std::vector<DimensionHandle> x_red, y_red;
        x_red.reserve(rank + 1);
        y_red.reserve(rank + 1);
        if (axis == 0)
        {
            x_red.push_back(C);
            y_red.push_back(K);
        }
        x_red.push_back(P);
        y_red.push_back(P);
        x_red.push_back(ctx->Dim(x, 1-axis));
        y_red.push_back(ctx->Dim(y, 1-axis));
        if (axis == 1)
        {
            x_red.push_back(C);
            y_red.push_back(K);
        }

        ctx->set_output(0, ctx->MakeShape({ C, K }));
        ctx->set_output(1, ctx->MakeShape(x_red));
        ctx->set_output(2, ctx->MakeShape(y_red));
    }
    else
    {
        ctx->set_output(0, ctx->UnknownShape());
        ctx->set_output(1, ctx->UnknownShape());
        ctx->set_output(2, ctx->UnknownShape());
    }
    return Status::OK();
}

REGISTER_OP("BlocksparseReducedDW")
    .Input("x:  n_params * half")
    .Input("dy: n_params * half")
    .Input("scale: float")        // scalar host tensor
    .Input("dwi: n_dwi * float")  // dw input to accumulate on top of
    .Output("dw: float")
    .Output("x_reduced:  half")
    .Output("dy_reduced: half")
    .Attr("n_params: int")
    .Attr("n_dwi: int >= 0")
    .Attr("bsize: int")
    .Attr("norm: int")
    .Attr("axis: int")
    .SetShapeFn(ReducedDWShape)
    .Doc(R"doc(
Block reduced full param gradient for use in network growth.
)doc");

bool BlocksparseFeatureReduceNC(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type);
bool BlocksparseFeatureReduceCN(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type);
bool hGemmTN(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale);
bool hGemmNT(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale);

class BlocksparseReducedDWOp : public OpKernel
{
public:
    explicit BlocksparseReducedDWOp(OpKernelConstruction* ctx) : OpKernel(ctx), major_version(0)
    {
        int bsize;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize", &bsize));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("norm",  &norm ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",  &axis ));
        OP_REQUIRES(ctx, axis == 0 || axis == 1, errors::InvalidArgument("invalid feature axis, only 0,1 supported."));
        if (axis == 0)
            OP_REQUIRES(ctx, bsize == 8 || bsize == 16 || bsize == 32, errors::InvalidArgument("Only feature axis=0 supports blocksizes: 8,16,32"));
        else
            OP_REQUIRES(ctx, bsize == 32 || bsize == 64, errors::InvalidArgument("Only feature axis=0 supports blocksizes: 32,64"));

        bshift = bsize == 8 ? 3 : bsize == 16 ? 4 : bsize == 32 ? 5 : 6;
    }
    void Compute(OpKernelContext* ctx) override
    {
        OpInputList x, y;
        ctx->input_list( "x", &x);
        ctx->input_list("dy", &y);
        uint params = x.size();
        float scale = ctx->input(params*2).scalar<float>()();
        OP_REQUIRES(ctx, params <= 8, errors::InvalidArgument("No more than 8 inputs allowed."));

        uint C  = x[0].dim_size(axis);
        uint K  = y[0].dim_size(axis);
        uint bC = C >> bshift;
        uint bK = K >> bshift;
        uint N  = x[0].dim_size(1-axis);
        TensorShape shapeX, shapeY;
        if (axis == 0)
        {
            shapeX.AddDim(bC);
            shapeY.AddDim(bK);
        }
        shapeX.AddDim(params);
        shapeY.AddDim(params);
        shapeX.AddDim(N);
        shapeY.AddDim(N);
        if (axis == 1)
        {
            shapeX.AddDim(bC);
            shapeY.AddDim(bK);
        }

        if (major_version == 0)
        {
            GetCountSMsVersion(&major_version, NULL);
            OP_REQUIRES(ctx, major_version >= 7, errors::InvalidArgument("Tensorcore GPU required"));

            OP_REQUIRES(ctx, (bC & 1) == 0 && (bK & 1) == 0, errors::InvalidArgument("Block reduced feature dim must be multiple of 2."));

            ClosestDivisorTo4(axis == 0 ? CEIL_DIV(bC, 32) : CEIL_DIV(bC, 64), true, &blk_a, &blk_A);
            ClosestDivisorTo4(axis == 0 ? CEIL_DIV(bK, 32) : CEIL_DIV(bK, 64),false, &blk_b, &blk_B);
        }

        struct Plist<ehalf,8> X, Y;
        for (int i = 0; i < params; ++i)
        {
            X.a[i] = (const ehalf*)x[i].flat<EHALF>().data();
            Y.a[i] = (const ehalf*)y[i].flat<EHALF>().data();
        }

        float* DW;
        uint accumulate;
        if (ctx->num_inputs() > params*2 + 1)
        {
            // accumulate to DW in place
            accumulate = 1;
            const Tensor& dw = ctx->input(params*2 + 1);
            ctx->set_output(0, dw);
            DW = (float*)dw.flat<float>().data();
        }
        else
        {
            accumulate = 0;
            Tensor *dw;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ bC, bK }), &dw));
            DW = dw->flat<float>().data();
        }
        Tensor *redX, *redY;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shapeX, &redX));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, shapeY, &redY));
        ehalf* RedX = (ehalf*)redX->flat<EHALF>().data();
        ehalf* RedY = (ehalf*)redY->flat<EHALF>().data();

        CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        if (scale != 0.0f)
        {
            if (axis == 0)
            {
                BlocksparseFeatureReduceCN(stream, RedX, &X, params, C, N, bshift, norm);
                BlocksparseFeatureReduceCN(stream, RedY, &Y, params, K, N, bshift, norm);
            }
            else
            {
                BlocksparseFeatureReduceNC(stream, RedX, &X, params, C, N, bshift, norm);
                BlocksparseFeatureReduceNC(stream, RedY, &Y, params, K, N, bshift, norm);
            }
        }
        if (axis == 0)
            hGemmNT(stream, RedX, RedY, DW, bC, bK, N*params, blk_A, blk_B, blk_a, blk_b, accumulate, scale);
        else
            hGemmTN(stream, RedX, RedY, DW, bC, bK, N*params, blk_A, blk_B, blk_a, blk_b, accumulate, scale);
    }
    int  bshift, norm, axis, major_version;
    uint blk_A, blk_B, blk_a, blk_b;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseReducedDW" ).Device(DEVICE_GPU).HostMemory("scale"),BlocksparseReducedDWOp);



