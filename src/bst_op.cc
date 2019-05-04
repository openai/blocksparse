
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "gpu_types.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using perftools::gputools::cuda::CUDAStream;

Status UnchangedShape(shape_inference::InferenceContext* ctx);


////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// BlocksparseTransformerOp //////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

Status nt_shape(InferenceContext* ctx)
{
    int heads, blocks, blk_size;
    TF_RETURN_IF_ERROR(ctx->GetAttr("heads",    &heads   ));
    TF_RETURN_IF_ERROR(ctx->GetAttr("blocks",   &blocks  ));
    TF_RETURN_IF_ERROR(ctx->GetAttr("blk_size", &blk_size));
    ShapeHandle x = ctx->input(0);

    if (ctx->RankKnown(x))
    {
      // (minbatch, heads, blocks, block_size, block_size)
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(x, 0),
        ctx->MakeDim(heads),
        ctx->MakeDim(blocks),
        ctx->MakeDim(blk_size),
        ctx->MakeDim(blk_size)
      }));
    }
    else
      ctx->set_output(0, ctx->UnknownShape());

    return Status::OK();
}
Status xn_shape(InferenceContext* ctx)
{
    int ctx_blks_c, blk_size;
    TF_RETURN_IF_ERROR(ctx->GetAttr("ctx_blks_c", &ctx_blks_c));
    TF_RETURN_IF_ERROR(ctx->GetAttr("blk_size",   &blk_size));
    ShapeHandle b = ctx->input(1);

    if (ctx->RankKnown(b))
    {
      // (batch, ctx_size, state_size)
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(b, 0),
        ctx->MakeDim(ctx_blks_c * blk_size),
        ctx->Dim(b, 2)
      }));
    }
    else
      ctx->set_output(0, ctx->UnknownShape());
    return Status::OK();
}

REGISTER_OP("BlocksparseTransformerNT")
    .Input("a: T")
    .Input("b: T")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: CT")
    .Attr("T: {half, float}")
    .Attr("CT: {half, bfloat16}")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks_a: int")
    .Attr("ctx_blks_b: int")
    .Attr("ctx_blks_c: int = 0")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(nt_shape)
    .Doc(R"doc(
Multiply the dense matrix "a" by the dense matrix "b.T" and produce blocksparse output "c".
)doc");

REGISTER_OP("BlocksparseTransformerNN")
    .Input("a: AT")
    .Input("b: T")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: T")
    .Attr("T: {half, float}")
    .Attr("AT: {half, bfloat16}")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks_a: int = 0")
    .Attr("ctx_blks_b: int")
    .Attr("ctx_blks_c: int")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(xn_shape)
    .Doc(R"doc(
Multiply the blocksparse matrix "a" by the dense matrix "b" and produce dense output "c".
)doc");

REGISTER_OP("BlocksparseTransformerTN")
    .Input("a: AT")
    .Input("b: T")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: T")
    .Attr("T: {half, float}")
    .Attr("AT: {half, bfloat16}")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks_a: int = 0")
    .Attr("ctx_blks_b: int")
    .Attr("ctx_blks_c: int")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(xn_shape)
    .Doc(R"doc(
Multiply the blocksparse matrix "a.T" by the dense matrix "b" and produce dense output "c".
)doc");

template <typename CT, typename CV2, typename CV4>
bool bst_hgemm_nt(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b,    CT* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim);
bool bst_hgemm_xn(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b, ehalf* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_b, uint ctx_blks_c, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_blks);

bool bst_sgemm_nt(CUstream stream, const uint2* lut, const float* a, const float* b, bhalf* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_a, uint ctx_blks_b, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim);
bool bst_sgemm_xn(CUstream stream, const uint2* lut, const bhalf* a, const float* b, float* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks_b, uint ctx_blks_c, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_blks);

template <uint OP>
class BlocksparseTransformerOp : public OpKernel {
 public:
  explicit BlocksparseTransformerOp(OpKernelConstruction* ctx) : OpKernel(ctx), major_(0), magic_(0), shift_(0), head_state_(0)
  {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("heads",      &heads_      ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",     &blocks_     ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size",   &blk_size_   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks_a", &ctx_blks_a_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks_b", &ctx_blks_b_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks_c", &ctx_blks_c_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("nn_max",     &nn_max_     ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tn_max",     &tn_max_     ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",      &bench_      ));
        repeat_ = bench_ ? bench_ : 1;
        flops_ = (float)(blocks_ * blk_size_ * blk_size_);
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (major_ == 0)
    {
      GetCountSMsVersion(&major_, NULL);
      if (bench_)
      {
        const char* op; // =  ? "NT" : OP == NN_OP ? "NN" : "TN";
        int ctx_blks_q, ctx_blks_k;
             if (OP == NT_OP) { ctx_blks_q = ctx_blks_a_; ctx_blks_k = ctx_blks_b_; op = "NT"; }
        else if (OP == NN_OP) { ctx_blks_q = ctx_blks_c_; ctx_blks_k = ctx_blks_b_; op = "NN"; }
        else                  { ctx_blks_q = ctx_blks_b_; ctx_blks_k = ctx_blks_c_; op = "TN"; }
        sprintf(bench_string_, "op:%s bsize:%02dx%02d blocks:%6d ctx:%5dq%5dk", op, blk_size_, blk_size_, blocks_, ctx_blks_q, ctx_blks_k);
      }
    }
    if (OP == NT_OP)
      this->Compute_NT(ctx);
    else
      this->Compute_XN(ctx, OP, OP == NN_OP ? nn_max_ : tn_max_);
  }
  void Compute_NT(OpKernelContext* ctx)
  {
    const Tensor& a   = ctx->input(0);
    const Tensor& b   = ctx->input(1);
    const Tensor& lut = ctx->input(2);

    OP_REQUIRES(ctx, a.dims() == 3 && b.dims() == 3, errors::InvalidArgument("Mismatched Shapes: a,b"));
    OP_REQUIRES(ctx, lut.dims() == 3,                errors::InvalidArgument("bad lut"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = a.dim_size(0);
    uint state_dim = a.dim_size(2);

    if (head_state_ == 0)
    {
      OP_REQUIRES(ctx,
        b.dim_size(0) == batch_dim &&
        b.dim_size(2) == state_dim, errors::InvalidArgument("Mismatched Shapes"));

      OP_REQUIRES(ctx, a.dim_size(1) == ctx_blks_a_ * blk_size_, errors::InvalidArgument("Bad A context length"));
      OP_REQUIRES(ctx, b.dim_size(1) == ctx_blks_b_ * blk_size_, errors::InvalidArgument("Bad B context length"));

      head_state_ = state_dim / heads_;
      OP_REQUIRES(ctx, state_dim % heads_ == 0,               errors::InvalidArgument("state_dim not evenly divisible by number of heads"));
      OP_REQUIRES(ctx, (head_state_ & 7) == 0,                errors::InvalidArgument("Head state dim must be multiple of 8, and ideally a multiple of 64"));
      OP_REQUIRES(ctx, lut_heads == heads_ || lut_heads == 1, errors::InvalidArgument("Bad head dim"));
    }

    Tensor* c;
    TensorShape shapeC({batch_dim, heads_, blocks_, blk_size_, blk_size_});
    OP_REQUIRES(ctx, shapeC.num_elements() < (1ull << 32), errors::InvalidArgument("Attention space too large.  Only 32 bit address offsets currently supported (easily fixed if needed.)"));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shapeC, &c));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, bench_string_, 0, flops_ * (float)(batch_dim * state_dim), repeat_);

    if (a.dtype() == DT_HALF)
    {
      OP_REQUIRES(ctx, major_ >= 7, errors::InvalidArgument("Tensorcore GPU required"));

      const ehalf* a_ptr = (const ehalf*)a.tensor_data().data();
      const ehalf* b_ptr = (const ehalf*)b.tensor_data().data();

      for (int r = 0; r < repeat_; r++)
        if (c->dtype() == DT_HALF)
          bst_hgemm_nt<ehalf,ehalf2,ehalf4>(stream, l_ptr, a_ptr, b_ptr, (ehalf*)c->tensor_data().data(), blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, lut_dim);
        else
          bst_hgemm_nt<bhalf,bhalf2,bhalf4>(stream, l_ptr, a_ptr, b_ptr, (bhalf*)c->tensor_data().data(), blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, lut_dim);
    }
    else
    {
      const float* a_ptr = (const float*)a.tensor_data().data();
      const float* b_ptr = (const float*)b.tensor_data().data();
            bhalf* c_ptr = (bhalf*)c->tensor_data().data();

      OP_REQUIRES(ctx, blk_size_ == 32, errors::InvalidArgument("Only blocksize=32 supported for fp32 pathway."));

      for (int r = 0; r < repeat_; r++)
        bst_sgemm_nt(stream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_a_, ctx_blks_b_, heads_, head_state_, lut_heads, lut_dim);
    }

    if (bench) delete bench;
  }
  void Compute_XN(OpKernelContext* ctx, uint op, uint max_lut)
  {
    const Tensor& a   = ctx->input(0);
    const Tensor& b   = ctx->input(1);
    const Tensor& lut = ctx->input(2 + op);

    OP_REQUIRES(ctx, a.dims() == 5 && b.dims() == 3, errors::InvalidArgument("Mismatched Shapes: a,b"));
    OP_REQUIRES(ctx, lut.dims() == 3,                errors::InvalidArgument("Bad lut"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = b.dim_size(0);
    uint state_dim = b.dim_size(2);

    if (head_state_ == 0)
    {
      OP_REQUIRES(ctx,
        a.dim_size(0) == batch_dim &&
        a.dim_size(1) == heads_    &&
        a.dim_size(2) == blocks_   &&
        a.dim_size(3) == blk_size_ &&
        a.dim_size(4) == blk_size_, errors::InvalidArgument("Mismatched A shape"));

      head_state_ = state_dim / heads_;
      OP_REQUIRES(ctx, state_dim % heads_ == 0,                  errors::InvalidArgument("state_dim not evenly divisible by number of heads"));
      OP_REQUIRES(ctx, (head_state_ & 7) == 0,                   errors::InvalidArgument("Head state dim must be multiple of 8, and ideally a multiple of 64"));
      OP_REQUIRES(ctx, b.dim_size(1) == ctx_blks_b_ * blk_size_, errors::InvalidArgument("Bad B context length"));
      OP_REQUIRES(ctx, lut_heads == heads_ || lut_heads == 1,    errors::InvalidArgument("Bad head dim"));

      uint div = CEIL_DIV(head_state_, 64);
      magicu64(div, magic_, shift_);
      OP_REQUIRES(ctx, magic_ > 0, errors::InvalidArgument("Bad magic for div: ", div));
    }

    Tensor* c;
    TensorShape shapeC({batch_dim, ctx_blks_c_ * blk_size_, state_dim});
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shapeC, &c));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, bench_string_, 0, flops_ * (float)(batch_dim * state_dim), repeat_);

    if (a.dtype() == DT_HALF)
    {
      OP_REQUIRES(ctx, major_ >= 7, errors::InvalidArgument("Tensorcore GPU required"));

      const ehalf* a_ptr = (const ehalf*)a.tensor_data().data();
      const ehalf* b_ptr = (const ehalf*)b.tensor_data().data();
            ehalf* c_ptr = (      ehalf*)c->tensor_data().data();

      for (int r = 0; r < repeat_; r++)
        bst_hgemm_xn(stream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, lut_dim, op, magic_, shift_, max_lut);
    }
    else
    {
      const bhalf* a_ptr = (const bhalf*)a.tensor_data().data();
      const float* b_ptr = (const float*)b.tensor_data().data();
            float* c_ptr = (      float*)c->tensor_data().data();

      OP_REQUIRES(ctx, blk_size_ == 32, errors::InvalidArgument("Only blocksize=32 supported for fp32 pathway."));

      for (int r = 0; r < repeat_; r++)
        bst_sgemm_xn(stream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_b_, ctx_blks_c_, heads_, head_state_, lut_heads, lut_dim, op, magic_, shift_, max_lut);
    }

    if (bench) delete bench;
  }
  int major_, heads_, blocks_, blk_size_, ctx_blks_a_, ctx_blks_b_, ctx_blks_c_, nn_max_, tn_max_, bench_, repeat_, flops_;
  uint magic_, shift_, head_state_;
  char bench_string_[256];
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerNT").Device(DEVICE_GPU),BlocksparseTransformerOp<NT_OP>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerNN").Device(DEVICE_GPU),BlocksparseTransformerOp<NN_OP>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerTN").Device(DEVICE_GPU),BlocksparseTransformerOp<TN_OP>);


template <typename T, typename V>
bool BlocksparseMaskedSoftmax(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const bhalf* x,
              T* y,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale);

REGISTER_OP("BlocksparseMaskedSoftmax")
    .Input("x: bfloat16")
    .Input("scale: float")
    .Input("lut: int32")
    .Input("mask: MT")
    .Output("y: T")
    .Attr("T: {half, bfloat16}")
    .Attr("MT: { uint8, uint16, uint32, uint64 }")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("lut_max: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Blocksparse softmax with mask
)doc");

REGISTER_OP("BlocksparseSoftmax")
    .Input("x: bfloat16")
    .Input("scale: float")
    .Input("lut: int32")
    .Output("y: T")
    .Attr("T: {half, bfloat16}")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("lut_max: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Blocksparse softmax without mask
)doc");

class BlocksparseMaskedSoftmaxOp : public OpKernel {
 public:
  explicit BlocksparseMaskedSoftmaxOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &blocks_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size", &blk_size_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks", &ctx_blks_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lut_max",  &lut_max_  ));

    OP_REQUIRES(ctx, lut_max_ * blk_size_ <= 32*1024, errors::InvalidArgument("max sparse sofmax dim: 32K"));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& s   = ctx->input(1);
    const Tensor& lut = ctx->input(2);

    OP_REQUIRES(ctx, x.dims()   == 5, errors::InvalidArgument("expecting 5 dims: (batch, head, block, blk_size, blk_size)"));
    OP_REQUIRES(ctx, lut.dims() == 3, errors::InvalidArgument("expecting 3 lut dims (head, entry, data)"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = x.dim_size(0);
    uint head_dim  = x.dim_size(1);

    uint mask_heads = 1;
    const char* m_ptr = NULL;
    if (ctx->num_inputs() > 3)
    {
      const Tensor& mask = ctx->input(3);
      OP_REQUIRES(ctx, mask.dims() == 3, errors::InvalidArgument("expecting 3 mask dims (head, blk_size, block)"));
      mask_heads = mask.dim_size(0);
      m_ptr = mask.tensor_data().data();
    }
    OP_REQUIRES(ctx, lut_heads  == head_dim || lut_heads  == 1, errors::InvalidArgument("Bad lut head dim"));
    OP_REQUIRES(ctx, mask_heads == head_dim || mask_heads == 1, errors::InvalidArgument("Bad mask head dim"));

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();
    const bhalf* x_ptr = (const bhalf*)x.tensor_data().data();
    float scale = s.scalar<float>()();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    if (y->dtype() == DT_HALF)
      BlocksparseMaskedSoftmax<ehalf,ehalf2>(stream, l_ptr, m_ptr, x_ptr, (ehalf*)y->tensor_data().data(), blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, mask_heads, scale);
    else
      BlocksparseMaskedSoftmax<bhalf,bhalf2>(stream, l_ptr, m_ptr, x_ptr, (bhalf*)y->tensor_data().data(), blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, mask_heads, scale);
  }
  int blocks_, blk_size_, ctx_blks_, lut_max_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmax").Device(DEVICE_GPU).HostMemory("scale"),BlocksparseMaskedSoftmaxOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseSoftmax"      ).Device(DEVICE_GPU).HostMemory("scale"),BlocksparseMaskedSoftmaxOp);


template <typename T, typename V>
bool BlocksparseMaskedSoftmaxGrad(CUstream stream,
    const uint2* lut,
    const     T* dy,
    const     T* y,
              T* dx,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    float scale);


REGISTER_OP("BlocksparseSoftmaxGrad")
    .Input("dy: T")
    .Input("y: T")
    .Input("scale: float")
    .Input("lut: int32")
    .Output("dx: T")
    .Attr("T: {half, bfloat16}")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("lut_max: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Blocksparse softmax grad
)doc");

class BlocksparseMaskedSoftmaxGradOp : public OpKernel {
 public:
  explicit BlocksparseMaskedSoftmaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &blocks_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size", &blk_size_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks", &ctx_blks_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lut_max",  &lut_max_  ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy  = ctx->input(0);
    const Tensor& y   = ctx->input(1);
    const Tensor& s   = ctx->input(2);
    const Tensor& lut = ctx->input(3);

    OP_REQUIRES(ctx, dy.dims()  == 5, errors::InvalidArgument("expecting 5 dims: (batch, head, block, blk_size, blk_size)"));
    OP_REQUIRES(ctx, lut.dims() == 3, errors::InvalidArgument("expecting 3 lut dims (head, entry, data)"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = dy.dim_size(0);
    uint head_dim  = dy.dim_size(1);

    OP_REQUIRES(ctx, lut_heads  == head_dim || lut_heads  == 1, errors::InvalidArgument("Bad lut head dim"));

    Tensor* dx = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();
    float scale = s.scalar<float>()();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    if (dy.dtype() == DT_HALF)
    {
      const ehalf* dy_ptr = (const ehalf*)dy.tensor_data().data();
      const ehalf*  y_ptr = (const ehalf*)y.tensor_data().data();
            ehalf* dx_ptr = (      ehalf*)dx->tensor_data().data();

      BlocksparseMaskedSoftmaxGrad<ehalf,ehalf2>(stream, l_ptr, dy_ptr, y_ptr, dx_ptr, blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, scale);
    }
    else
    {
      const bhalf* dy_ptr = (const bhalf*)dy.tensor_data().data();
      const bhalf*  y_ptr = (const bhalf*)y.tensor_data().data();
            bhalf* dx_ptr = (      bhalf*)dx->tensor_data().data();

      BlocksparseMaskedSoftmaxGrad<bhalf,bhalf2>(stream, l_ptr, dy_ptr, y_ptr, dx_ptr, blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, scale);
    }
  }
  int blocks_, blk_size_, ctx_blks_, lut_max_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale"),BlocksparseMaskedSoftmaxGradOp);


bool BstPartialAutoregressiveMask(CUstream stream,
    const int2* lut, const char* maskI, char* maskO,
    uint block_size, uint blocks, uint lut_heads, uint lut_dim, int autoregress_at_k);

REGISTER_OP("BstPartialAutoregressiveMask")
    .Input("mask: MT")
    .Input("lut: int32")
    .Input("autoregress_at_k: KT")
    .Output("mask_out: MT")
    .Attr("MT: { uint8, uint16, uint32, uint64 }")
    .Attr("KT: { int32, int64 }")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks_k: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Dynamically apply autoregressive property to softmax mask from autoregress_at_k on
)doc");

class PartialAutoregressiveMaskOp : public OpKernel {
 public:
  explicit PartialAutoregressiveMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    int ctx_blks_k;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",     &blocks_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size",   &blk_size_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks_k", &ctx_blks_k));
    ctx_keys_ = ctx_blks_k * blk_size_;
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& mask = ctx->input(0);
    const Tensor& lut  = ctx->input(1);
    const Tensor& k    = ctx->input(2);

    int key = k.dtype() == DT_INT64 ? (int)k.scalar<int64>()() : k.scalar<int32>()();

    OP_REQUIRES(ctx, key >= 0 && key < ctx_keys_, errors::InvalidArgument("autoregress_at_key out of range"));
    OP_REQUIRES(ctx, lut.dims()  == 3, errors::InvalidArgument("expecting 3 lut dims (head, entry, data)"));
    OP_REQUIRES(ctx, mask.dims() == 3, errors::InvalidArgument("expecting 3 mask dims (head, blk_size, block)"));

    uint mask_heads = mask.dim_size(0);
    uint lut_heads  = lut.dim_size(0);
    uint lut_dim    = lut.dim_size(1);

    OP_REQUIRES(ctx, mask_heads == lut_heads, errors::InvalidArgument("Bad lut/mask head dim"));

    Tensor* out = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mask.shape(), &out));

    const  int2* l_ptr = (const int2*)lut.flat<int32>().data();
    const  char* m_ptr = mask.tensor_data().data();
           char* o_ptr = (char*)out->tensor_data().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BstPartialAutoregressiveMask(stream, l_ptr, m_ptr, o_ptr, blk_size_, blocks_, lut_heads, lut_dim, key);
  }
  int blocks_, blk_size_, ctx_keys_;
};
REGISTER_KERNEL_BUILDER(Name("BstPartialAutoregressiveMask").Device(DEVICE_GPU).HostMemory("autoregress_at_k"),PartialAutoregressiveMaskOp);