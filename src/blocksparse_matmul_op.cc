
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/platform/stream_executor.h"
// #include "tensorflow/stream_executor/cuda/cuda_stream.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
// using perftools::gputools::cuda::AsCUDAStreamValue;

#include "gpu_types.h"
#include "blocksparse_matmul.h"

//static int first = 0;

template <MTYPE3(TA,TB,TC)>
class BlocksparseMatmulOp : public OpKernel
{
public:
    explicit BlocksparseMatmulOp(OpKernelConstruction* ctx, const char* op) : OpKernel(ctx), bsmm_(0)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("segments", &params_.segments));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("locks",    &params_.locks   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &params_.blocks  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bshift",   &params_.bshift  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("C",        &params_.C       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("K",        &params_.K       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shared",   &params_.shared  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",    &params_.alpha   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta",     &params_.beta    ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_));
        repeat_ = bench_ ? bench_ : 1;
        flops_ = (float)(params_.blocks << (params_.bshift*2));
        params_.pcount = 1;

        is_gpu_ = ctx->device_type() == DEVICE_GPU;

        // TODO: pack larger values of K in gridZ
        OP_REQUIRES(ctx, params_.K < (1<<(16+params_.bshift)), errors::InvalidArgument("K < 2**(16+bshift)"));
        OP_REQUIRES(ctx, params_.C < (1<<(16+params_.bshift)), errors::InvalidArgument("C < 2**(16+bshift)"));

        sprintf(bench_string_, "%s %02d-%d %05d", op, (1<<params_.bshift), axis_, params_.K);

        // sprintf(bench_string_, "BlocksparseMatmul op:%s axis:%d bshift:%d blocks:%5d segments:%5d locks:%5d C:%5d K%5d share:%4d repeat:%d",
        //     op, axis_, params_.bshift, params_.blocks, params_.segments, params_.locks, params_.C, params_.K, params_.shared, repeat_);

        // if (!first)
        // {
        //     printf("\n\n--------------f8-x10-a16-------------------\n\n");
        //     first = 1;
        // }
    }
    virtual ~BlocksparseMatmulOp() { delete bsmm_; }

    virtual void Compute(OpKernelContext* ctx) =0;

    Status Xprop_Compute(OpKernelContext* ctx)
    {
        const Tensor& A = ctx->input(0);
        const Tensor& B = ctx->input(1);
        const Tensor& L = ctx->input(2);

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

        Tensor* C;
        Status s = ctx->allocate_output(0, shapeC, &C);
        if (!s.ok()) return s;

        Tensor* Lock;
        int gridN64 = (N >> 6) + ((N & 63) != 0);
        s = ctx->allocate_output(1, TensorShape({gridN64 * params_.locks * 2}), &Lock);
        if (!s.ok()) return s;
        if (params_.locks > 0)
            params_.Lock = Lock->flat<int32>().data();
        else
            params_.Lock = nullptr;

        // Tensor* Scratch;
        // if (sizeof(TC) != 2 || (N & 7) != 0)
        //     gridN64 = 0;
        // int scratch_size = (gridN64 * params_.locks) << (6 + params_.bshift);
        // s = ctx->allocate_output(2, TensorShape({ scratch_size }), &Scratch);
        // if (!s.ok()) return s;
        // if (scratch_size > 0)
        //     params_.Scratch = (float4*)Scratch->flat<float>().data();
        // else
        //     params_.Scratch = nullptr;

        params_.N   = N;
        params_.Lut = L.flat<int32>().data();

              TC1* pC = (      TC1*)C->flat<TC>().data();
        const TA1* pA = (const TA1*)A.flat<TA>().data();
        const TB1* pB = (const TB1*)B.flat<TB>().data();

        if (is_gpu_)
            params_.stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

        return this->Op_Compute(pA, pB, pC);
    }
    Status Updat_Compute(OpKernelContext* ctx)
    {
        OpInputList x, dy;

        ctx->input_list( "x",  &x);
        ctx->input_list("dy", &dy);

        params_.pcount = x.size();

        if (params_.pcount > 8)
            return errors::Internal("No more than 8 inputs allowed.");

        struct plist8<TA1> X;
        struct plist8<TB1> DY;
        for (int i = 0; i < params_.pcount; ++i)
        {
            X.a[i]  = (const TA1*)x[i].flat<TA>().data();
            DY.a[i] = (const TB1*)dy[i].flat<TB>().data();
        }
        params_.N = 1;
        int rank = x[0].dims();
        for (int i = 0; i < rank; i++)
            if (i != axis_)
                params_.N *= x[0].dim_size(i);

        TC1* DW;
        if (params_.beta == 0.0f)
        {
            if (ctx->num_inputs() != params_.pcount*2 + 1)
                return errors::Internal("with beta=0.0, use BlocksparseMatmulDW ", ctx->num_inputs());

            int bsize = 1 << params_.bshift;
            TensorShape shapeC({ params_.blocks, bsize, bsize });
            Tensor* C;
            Status s = ctx->allocate_output(0, shapeC, &C);
            if (!s.ok()) return s;
            DW = (TC1*)C->flat<TC>().data();
        }
        else
        {
            if (ctx->num_inputs() != params_.pcount*2 + 2)
                return errors::Internal("with beta!=0.0, use BlocksparseMatmulDWA ", ctx->num_inputs());

            // accumulate to C in place
            const Tensor& C = ctx->input(params_.pcount*2 + 1);
            ctx->set_output(0, C);
            DW = (TC1*)C.flat<TC>().data();
        }
        params_.Lut = ctx->input(params_.pcount*2).flat<int32>().data();

        if (is_gpu_)
            params_.stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

        return this->Op_Compute((const TA1*)&X, (const TB1*)&DY, DW);
    }
    Status Op_Compute(const TA1* pA, const TB1* pB, TC1* pC)
    {
        Benchmark* bench = nullptr;
        if (bench_) bench = new Benchmark(bench_string_, 0, flops_ * params_.N * params_.pcount, repeat_, is_gpu_);

        Status status;
        for (int r = 0; r < repeat_; r++)
            status = bsmm_->Compute(pA, pB, pC);

        if (bench) delete bench;
        return status;
    }

    BlocksparseMatmul<OTYPE3(TA,TB,TC)>* bsmm_;

    bsmm_params params_;
    int   axis_, bench_, repeat_;
    float flops_;
    bool  is_gpu_;
    char  bench_string_[256];
};


template <MTYPE3(TA,TB,TC)>
class BlocksparseMatmulFpropOp : public BlocksparseMatmulOp<NTYPE3(TA,TB,TC)> {
 public:
    explicit BlocksparseMatmulFpropOp(OpKernelConstruction* ctx) : BlocksparseMatmulOp<NTYPE3(TA,TB,TC)>(ctx, "fprop")
    {
        if (this->axis_ == 0)
            this->bsmm_ = new BlocksparseMatmulFprop_CN<OTYPE3(TA,TB,TC)>(&this->params_);
        else
            this->bsmm_ = new BlocksparseMatmulFprop_NC<OTYPE3(TA,TB,TC)>(&this->params_);
    }
    void Compute(OpKernelContext* ctx) override
    {
        OP_REQUIRES_OK(ctx, this->Xprop_Compute(ctx));
    }
};

#if DINTEL_MKL
template <MTYPE3(TA,TB,TC)>
class BlocksparseMatmulCPUOp : public BlocksparseMatmulOp<NTYPE3(TA,TB,TC)> {
 public:
    explicit BlocksparseMatmulCPUOp(OpKernelConstruction* ctx) : BlocksparseMatmulOp<NTYPE3(TA,TB,TC)>(ctx, "fprop")
    {
        this->params_.locks = 0;

        if (this->params_.bshift == 5)
            this->bsmm_ = new BlocksparseMatmulFprop_CPU<32,OTYPE3(TA,TB,TC)>(&this->params_);
        else if (this->params_.bshift == 4)
            this->bsmm_ = new BlocksparseMatmulFprop_CPU<16,OTYPE3(TA,TB,TC)>(&this->params_);
        else
            this->bsmm_ = new BlocksparseMatmulFprop_CPU< 8,OTYPE3(TA,TB,TC)>(&this->params_);
    }
    void Compute(OpKernelContext* ctx) override
    {
        OP_REQUIRES_OK(ctx, this->Xprop_Compute(ctx));
    }
};
#endif // DINTEL_MKL

template <MTYPE3(TA,TB,TC)>
class BlocksparseMatmulBpropOp : public BlocksparseMatmulOp<NTYPE3(TA,TB,TC)> {
 public:
    explicit BlocksparseMatmulBpropOp(OpKernelConstruction* ctx) : BlocksparseMatmulOp<NTYPE3(TA,TB,TC)>(ctx, "bprop")
    {
        if (this->axis_ == 0)
            this->bsmm_ = new BlocksparseMatmulBprop_CN<OTYPE3(TA,TB,TC)>(&this->params_);
        else
            this->bsmm_ = new BlocksparseMatmulBprop_NC<OTYPE3(TA,TB,TC)>(&this->params_);
    }
    void Compute(OpKernelContext* ctx) override
    {
        OP_REQUIRES_OK(ctx, this->Xprop_Compute(ctx));
    }
};
template <MTYPE3(TA,TB,TC)>
class BlocksparseMatmulUpdatOp : public BlocksparseMatmulOp<NTYPE3(TA,TB,TC)> {
 public:
    explicit BlocksparseMatmulUpdatOp(OpKernelConstruction* ctx) : BlocksparseMatmulOp<NTYPE3(TA,TB,TC)>(ctx, "updat")
    {
        if (this->axis_ == 0)
            this->bsmm_ = new BlocksparseMatmulUpdat_CN<OTYPE3(TA,TB,TC)>(&this->params_);
        else
            this->bsmm_ = new BlocksparseMatmulUpdat_NC<OTYPE3(TA,TB,TC)>(&this->params_);
    }
    void Compute(OpKernelContext* ctx) override
    {
        OP_REQUIRES_OK(ctx, this->Updat_Compute(ctx));
    }
};

Status XpropShape(InferenceContext* ctx)
{
    int    K; TF_RETURN_IF_ERROR(ctx->GetAttr(   "K",    &K));
    int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

    // C ==> K
    ShapeHandle x = ctx->input(0);
    int rank = ctx->Rank(x);
    if (rank > 0)
    {
        std::vector<DimensionHandle> shape;
        shape.reserve(rank);
        for (int i = 0; i < rank; i++)
            shape.push_back(i == axis ? ctx->MakeDim(K) : ctx->Dim(x, i));

        ctx->set_output(0, ctx->MakeShape(shape));
    }
    return Status::OK();
}
Status UpdatShape(InferenceContext* ctx)
{
    int blocks, bshift;
    TF_RETURN_IF_ERROR(ctx->GetAttr("blocks", &blocks));
    TF_RETURN_IF_ERROR(ctx->GetAttr("bshift", &bshift));

    // (blocks, block_size, block_size)
    DimensionHandle bsize = ctx->MakeDim(1 << bshift);
    ctx->set_output(0, ctx->MakeShape({ ctx->MakeDim(blocks), bsize, bsize }));
    return Status::OK();
}

REGISTER_OP("BlocksparseMatmul")
    .Input("x: dtype_x")
    .Input("w: dtype_w")
    .Input("lut: int32")
    .Input("lut_dx: int32")
    .Input("lut_dw: int32")
    .Output("y: dtype_y")
    .Output("temp: int32")
    //.Output("temp2: float")
    .Attr("dtype_x: {half, float, bfloat16}")
    .Attr("dtype_w: {half, float, bfloat16}")
    .Attr("dtype_y: {half, float, bfloat16}")
    .Attr("dtype_dw: {half, float, bfloat16} = DT_FLOAT")
    .Attr("blocks: int >=0")
    .Attr("bshift: int = 5")
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
    .Attr("bench: int = 0")
    .SetShapeFn(XpropShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_x").TypeConstraint<FLOAT>("dtype_w").TypeConstraint<FLOAT>("dtype_y"),BlocksparseMatmulFpropOp<FLOAT2, FLOAT2, FLOAT2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<EHALF>("dtype_w").TypeConstraint<EHALF>("dtype_y"),BlocksparseMatmulFpropOp<EHALF2, EHALF2, EHALF2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<BHALF>("dtype_w").TypeConstraint<BHALF>("dtype_y"),BlocksparseMatmulFpropOp<BHALF2, BHALF2, BHALF2>);

#if DINTEL_MKL
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmul").Device(DEVICE_CPU).TypeConstraint<FLOAT>("dtype_x").TypeConstraint<FLOAT>("dtype_w").TypeConstraint<FLOAT>("dtype_y"),BlocksparseMatmulCPUOp<FLOAT2, FLOAT2, FLOAT2>);
#endif // DINTEL_MKL


REGISTER_OP("BlocksparseMatmulDX")
    .Input("dy: dtype_dy")
    .Input("w: dtype_w")
    .Input("lut: int32")
    .Output("dx: dtype_dx")
    .Output("temp: int32")
    //.Output("temp2: float")
    .Attr("dtype_dy: {half, float, bfloat16}")
    .Attr("dtype_w: {half, float, bfloat16}")
    .Attr("dtype_dx: {half, float, bfloat16}")
    .Attr("blocks: int >=0")
    .Attr("bshift: int = 5")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("bench: int = 0")
    .SetShapeFn(XpropShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_w").TypeConstraint<FLOAT>("dtype_dx"),BlocksparseMatmulBpropOp<FLOAT2, FLOAT2, FLOAT2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_dy").TypeConstraint<EHALF>("dtype_w").TypeConstraint<EHALF>("dtype_dx"),BlocksparseMatmulBpropOp<EHALF2, EHALF2, EHALF2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_dy").TypeConstraint<BHALF>("dtype_w").TypeConstraint<BHALF>("dtype_dx"),BlocksparseMatmulBpropOp<BHALF2, BHALF2, BHALF2>);

//REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<EHALF>("dtype_w").TypeConstraint<FLOAT>("dtype_dx"),BlocksparseMatmulBpropOp<FLOAT2, EHALF2, FLOAT2>);
//REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDX").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<BHALF>("dtype_w").TypeConstraint<FLOAT>("dtype_dx"),BlocksparseMatmulBpropOp<FLOAT2, BHALF2, FLOAT2>);


REGISTER_OP("BlocksparseMatmulDW")
    .Input("x: params * dtype_x")
    .Input("dy: params * dtype_dy")
    .Input("lut: int32")
    .Output("dw: dtype_dw")
    .Attr("dtype_x: {half, float, bfloat16}")
    .Attr("dtype_dy: {half, float, bfloat16}")
    .Attr("dtype_dw: {half, float, bfloat16} = DT_FLOAT")
    .Attr("params: int")
    .Attr("blocks: int >=0")
    .Attr("bshift: int = 5")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 0.0")
    .Attr("bench: int = 0")
    .SetShapeFn(UpdatShape)
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<FLOAT2, FLOAT2, FLOAT2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<EHALF>("dtype_dy").TypeConstraint<EHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, EHALF2, EHALF2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<BHALF>("dtype_dy").TypeConstraint<BHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, BHALF2, BHALF2>);

// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<EHALF>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, EHALF2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, FLOAT2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<EHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, FLOAT2, EHALF2>);

// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<BHALF>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, BHALF2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, FLOAT2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDW").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<BHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, FLOAT2, BHALF2>);


REGISTER_OP("BlocksparseMatmulDWA")
    .Input("x: params * dtype_x")
    .Input("dy: params * dtype_dy")
    .Input("lut: int32")
    .Input("dwi: dtype_dw")
    .Output("dw: dtype_dw")
    .Attr("dtype_x: {half, float, bfloat16}")
    .Attr("dtype_dy: {half, float, bfloat16}")
    .Attr("dtype_dw: {half, float, bfloat16} = DT_FLOAT")
    .Attr("params: int")
    .Attr("blocks: int >=0")
    .Attr("bshift: int = 5")
    .Attr("segments: int = 0")
    .Attr("locks: int = 0")
    .Attr("axis: int = 1")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("shared: int = 0")
    .Attr("alpha: float = 1.0")
    .Attr("beta: float = 1.0")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(ctx->num_inputs()-1) );
      return Status::OK();
    })
    .Doc(R"doc(
Multiply the matrix "a" by the blocksparse matrix "b" and accumulate to "c".
)doc");

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<FLOAT>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<FLOAT2, FLOAT2, FLOAT2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<EHALF>("dtype_dy").TypeConstraint<EHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, EHALF2, EHALF2>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<BHALF>("dtype_dy").TypeConstraint<BHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, BHALF2, BHALF2>);

// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<EHALF>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, EHALF2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, FLOAT2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<EHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<EHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<EHALF2, FLOAT2, EHALF2>);

// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<BHALF>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, BHALF2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<FLOAT>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, FLOAT2, FLOAT2>);
// REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulDWA").Device(DEVICE_GPU).TypeConstraint<BHALF>("dtype_x").TypeConstraint<FLOAT>("dtype_dy").TypeConstraint<BHALF>("dtype_dw"),BlocksparseMatmulUpdatOp<BHALF2, FLOAT2, BHALF2>);




bool IdentityInitCK(CUstream stream, float* W, const int* lut, int CB, int KB, int blocks, int bshift);

REGISTER_OP("BlocksparseMatmulIdentityInit")
    .Input("lut: int32")
    .Output("w: float")
    .Attr("CB: int >=0")
    .Attr("KB: int >=0")
    .Attr("blocks: int >=0")
    .Attr("bshift: int >=0")
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bshift", &bshift_));
    bsize_ = 1 << bshift_;
  }

  void Compute(OpKernelContext* ctx) override {

    TensorShape c_shape({ blocks_, bsize_, bsize_ });

    Tensor* w = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &w));

        float*   w_ptr = w->flat<float>().data();
    const int* lut_ptr = ctx->input(0).flat<int32>().data();

    CUstream cu_stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    IdentityInitCK(cu_stream, w_ptr, lut_ptr, CB_, KB_, blocks_, bshift_);
  }
 private:

  int blocks_, bshift_, bsize_, CB_, KB_;
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseMatmulIdentityInit").Device(DEVICE_GPU), BlocksparseMatmulIdentityInitOp);


  // static void ClosestDivisorTo4(unsigned val, bool isA, unsigned* div, unsigned* res) {
  //        if ((val & 3) == 0) { *div = 4; *res = val >> 2; }
  //   else if ((val % 3) == 0) { *div = 3; *res = val  / 3; }
  //   else if ((val % 5) == 0) { *div = 5; *res = val  / 5; }
  //   else if ((val & 1) == 0) { *div = 2; *res = val >> 1; }
  //   else if ((val % 7) == 0) { *div = 7; *res = val  / 7; }
  //   else if (isA) { *div = val; *res =   1; }
  //   else          { *div = 1;   *res = val; }
  // }