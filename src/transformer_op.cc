
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

template <typename T> bool TopK(CUstream stream, T* y, uint* a, const T* x, uint topK, uint N, uint K, uint rebase);

REGISTER_OP("Topk")
    .Input("x: T")
    .Input("k: int32")
    .Output("y: T")
    .Output("a: int32")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {

      ShapeHandle x = ctx->input(0);
      DimensionHandle k_dim;
      TF_RETURN_IF_ERROR(ctx->MakeDimForScalarInput(1, &k_dim));

      int rank = ctx->Rank(x);
      std::vector<DimensionHandle> dims;
      for (int i = 0; i < rank-1; ++i)
          dims.emplace_back(ctx->Dim(x, i));

      dims.emplace_back(k_dim);

      ShapeHandle s = ctx->MakeShape(dims);
      ctx->set_output(0, s);
      ctx->set_output(1, s);
      return Status::OK();
    })
    .Doc(R"doc(
Finds values and indices of the k largest entries for the last dimension.
)doc");

template <typename T, typename V>
class TopkOp : public OpKernel {
 public:
  explicit TopkOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& k = ctx->input(1);

    uint topK = k.scalar<int32>()();
    uint rank = x.dims();
    uint K    = x.dim_size(--rank);
    uint N    = 1;
    TensorShape shape;
    while (rank > 0)
    {
      uint dim = x.dim_size(--rank);
      N *= dim;
      shape.AddDim(dim);
    }
    shape.AddDim(topK);

    Tensor* y = NULL;
    Tensor* a = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shape, &a));

          V* y_ptr = (V*)y->flat<T>().data();
       uint* a_ptr = (uint*)a->flat<int32>().data();
    const V* x_ptr = (const V*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    TopK<V>(stream, y_ptr, a_ptr, x_ptr, topK, N, K, false);
  }
};
REGISTER_KERNEL_BUILDER(Name("Topk").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("k"),TopkOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Topk").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("k"),TopkOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("Topk").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("k"),TopkOp<BHALF,bhalf>);


REGISTER_OP("RectifiedTopK")
    .Input("x: T")
    .Input("k: int32")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("rebase: bool = true")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Finds values and indices of the k largest entries for the last dimension.
)doc");

template <typename T, typename V>
class RectifiedTopKOp : public OpKernel {
 public:
  explicit RectifiedTopKOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rebase", &rebase_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& k = ctx->input(1);

    uint topK = k.scalar<int32>()();
    uint rank = x.dims();
    uint K    = x.dim_size(--rank);
    uint N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

          V* y_ptr = (V*)y->flat<T>().data();
    const V* x_ptr = (const V*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    TopK<V>(stream, y_ptr, NULL, x_ptr, topK, N, K, rebase_);
  }
  bool rebase_;
};
REGISTER_KERNEL_BUILDER(Name("RectifiedTopK").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("k"),RectifiedTopKOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("RectifiedTopK").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("k"),RectifiedTopKOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("RectifiedTopK").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("k"),RectifiedTopKOp<BHALF,bhalf>);



template <typename T> bool MaskedTopKSoftmax(CUstream stream, T* y, const float* m, const T* x, uint topK, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

REGISTER_OP("MaskedTopKSoftmax")
    .Input("x: T")
    .Input("k: int32")
    .Input("scale: float")
    .Input("mask: n_mask * float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("n_mask: int >= 0")
    .Attr("bench: int = 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Finds values and indices of the k largest entries for the last dimension.
)doc");

template <typename T, typename V>
class MaskedTopKSoftmaxOp : public OpKernel {
 public:
  explicit MaskedTopKSoftmaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& k = ctx->input(1);
    const Tensor& s = ctx->input(2);
    OpInputList m;    ctx->input_list("mask", &m);

    // x: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    int rank = x.dims();
    uint D3 = x.dim_size(--rank);
    uint D2 = 1, D1 = 1, D0 = 1, M2 = 0, M1 = 0;
    const float* m_ptr = NULL;
    if (m.size() > 0)
    {
      // gather inner dimensions and strides of the mask
      // only dims 1 and 2 are unknown for the mask, so just deterimine those strides
      if (rank > 0) { D2 = x.dim_size(--rank); M2 = m[0].dim_size(rank) == 1 ? 0 : D3;    }
      if (rank > 0) { D1 = x.dim_size(--rank); M1 = m[0].dim_size(rank) == 1 ? 0 : D3*D2; }

      m_ptr = m[0].flat<float>().data();
    }
    while (rank > 0) { D0 *= x.dim_size(--rank); }

    OP_REQUIRES(ctx, D3 <= 1024, errors::Internal("D3 <= 1024: ", D3));
    OP_REQUIRES(ctx, D2 < 65536, errors::Internal("D2 < 65536: ", D2));
    OP_REQUIRES(ctx, D1 < 65536, errors::Internal("D1 < 65536: ", D1));

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

           V* y_ptr = (      V*)y->flat<T>().data();
    const  V* x_ptr = (const V*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    MaskedTopKSoftmax<V>(stream, y_ptr, m_ptr, x_ptr, k.scalar<int32>()(), D0, D1, D2, D3, M1, M2, s.scalar<float>()());
  }
};
REGISTER_KERNEL_BUILDER(Name("MaskedTopKSoftmax").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("k").HostMemory("scale"),MaskedTopKSoftmaxOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("MaskedTopKSoftmax").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("k").HostMemory("scale"),MaskedTopKSoftmaxOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("MaskedTopKSoftmax").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("k").HostMemory("scale"),MaskedTopKSoftmaxOp<BHALF,bhalf>);


template <typename T> bool MaskedSoftmax(CUstream stream, T* y, const T* x, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

REGISTER_OP("MaskedSoftmax")
    .Input("x: T")
    .Input("scale: float")
    .Input("mask: n_mask * float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("n_mask: int >= 0")
    .Attr("bench: int = 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
softmax with optional mask broadcast
)doc");

template <typename T, typename V>
class MaskedSoftmaxOp : public OpKernel {
 public:
  explicit MaskedSoftmaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench", &bench_ ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& s = ctx->input(1);
    OpInputList m;    ctx->input_list("mask", &m);

    // y: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    int rank = x.dims();
    uint D3 = x.dim_size(--rank);
    uint D2 = 1, D1 = 1, D0 = 1, M2 = 0, M1 = 0;
    const float* m_ptr = NULL;
    if (m.size() > 0)
    {
      // gather inner dimensions and strides of the mask
      // only dims 1 and 2 are unknown for the mask, so just deterimine those strides
      if (rank > 0) { D2 = x.dim_size(--rank); M2 = m[0].dim_size(rank) == 1 ? 0 : D3;    }
      if (rank > 0) { D1 = x.dim_size(--rank); M1 = m[0].dim_size(rank) == 1 ? 0 : D3*D2; }

      m_ptr = m[0].flat<float>().data();
    }
    while (rank > 0) { D0 *= x.dim_size(--rank); }

    OP_REQUIRES(ctx, D2 < 65536, errors::Internal("D2 < 65536: ", D2));
    OP_REQUIRES(ctx, D1 < 65536, errors::Internal("D1 < 65536: ", D1));

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

          V* y_ptr = (      V*)y->flat<T>().data();
    const V* x_ptr = (const V*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "MaskedSoftmax     (%6d,%4d,%4d,%4d) %d, %d", D0, D1, D2, D3, (uint)m.size(), (uint)sizeof(V));
      bench = new Benchmark(stream, bench_string, x.NumElements()*2*sizeof(V), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      MaskedSoftmax<V>(stream, y_ptr, x_ptr, m_ptr, D0, D1, D2, D3, M1, M2, s.scalar<float>()());

    if (bench) delete bench;
  }
  int bench_;
};
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmax").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("scale"),MaskedSoftmaxOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmax").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("scale"),MaskedSoftmaxOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmax").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("scale"),MaskedSoftmaxOp<BHALF,bhalf>);


template <typename T> bool MaskedSoftmaxGrad(CUstream stream, T* dx, const T* dy, const T* y, const float* m, uint D0, uint D1, uint D2, uint D3, uint M1, uint M2, float scale);

REGISTER_OP("MaskedSoftmaxGrad")
    .Input("dy: T")
    .Input("y: T")
    .Input("scale: float")
    .Input("mask: n_mask * float")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("n_mask: int >= 0")
    .Attr("bench: int = 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
softmax with optional mask broadcast gradient
)doc");

template <typename T, typename V>
class MaskedSoftmaxGradOp : public OpKernel {
 public:
  explicit MaskedSoftmaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench", &bench_ ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor&  y = ctx->input(1);
    const Tensor&  s = ctx->input(2);
    OpInputList m;     ctx->input_list("mask", &m);

    // y: D0, D1, D2, D3
    // m:  1, D1, D2, D3
    // m:  1,  1, D2, D3
    // m:  1,  1,  1, D3
    int rank = y.dims();
    uint D3 = y.dim_size(--rank);
    uint D2 = 1, D1 = 1, D0 = 1, M2 = 0, M1 = 0;
    const float* m_ptr = NULL;
    if (m.size() > 0)
    {
      // gather inner dimensions and strides of the mask
      // only dims 1 and 2 are unknown for the mask, so just deterimine those strides
      if (rank > 0) { D2 = y.dim_size(--rank); M2 = m[0].dim_size(rank) == 1 ? 0 : D3;    }
      if (rank > 0) { D1 = y.dim_size(--rank); M1 = m[0].dim_size(rank) == 1 ? 0 : D3*D2; }

      m_ptr = m[0].flat<float>().data();
    }
    while (rank > 0) { D0 *= y.dim_size(--rank); }

    OP_REQUIRES(ctx, D2 < 65536, errors::Internal("D2 < 65536: ", D2));
    OP_REQUIRES(ctx, D1 < 65536, errors::Internal("D1 < 65536: ", D1));

    Tensor* dx = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));

          V* dx_ptr = (      V*)dx->flat<T>().data();
    const V* dy_ptr = (const V*)dy.flat<T>().data();
    const V*  y_ptr = (const V*)y.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "MaskedSoftmaxGrad (%6d,%4d,%4d,%4d) %d, %d", D0, D1, D2, D3, (uint)m.size(), (uint)sizeof(V));
      bench = new Benchmark(stream, bench_string, dy.NumElements()*3*sizeof(V), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      MaskedSoftmaxGrad<V>(stream, dx_ptr, dy_ptr, y_ptr, m_ptr, D0, D1, D2, D3, M1, M2, s.scalar<float>()());

    if (bench) delete bench;
  }
  int bench_;
};
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("scale"),MaskedSoftmaxGradOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("scale"),MaskedSoftmaxGradOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("MaskedSoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("scale"),MaskedSoftmaxGradOp<BHALF,bhalf>);


template <typename T> bool Transpose_0213(CUstream stream, T* y, const T* x, uint D0, uint D1, uint D2, uint D3);

REGISTER_OP("Transpose0213")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 4, &x));
      ctx->set_output(0, ctx->MakeShape({ ctx->Dim(x,0), ctx->Dim(x,2), ctx->Dim(x,1), ctx->Dim(x,3) }));
      return Status::OK();
    })
    .Doc(R"doc(
Transpose op commonly used in transformer models (0,2,1,3).
)doc");

template <typename T, typename V>
class Transpose0213Op : public OpKernel {
 public:
  explicit Transpose0213Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    OP_REQUIRES(ctx, x.dims() == 4, errors::Internal("x.dims() == 4: ", x.dims()));

    uint D0 = x.dim_size(0);
    uint D1 = x.dim_size(1);
    uint D2 = x.dim_size(2);
    uint D3 = x.dim_size(3);

    OP_REQUIRES(ctx, D0 < 65536, errors::Internal("D0 < 65536: ", D0));
    OP_REQUIRES(ctx, D1 < 65536, errors::Internal("D1 < 65536: ", D1));

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({D0, D2, D1, D3}), &y));

          V* y_ptr = (      V*)y->flat<T>().data();
    const V* x_ptr = (const V*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Transpose_0213<V>(stream, y_ptr, x_ptr, D0, D1, D2, D3);

  }
};
REGISTER_KERNEL_BUILDER(Name("Transpose0213").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),Transpose0213Op<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Transpose0213").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),Transpose0213Op<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("Transpose0213").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),Transpose0213Op<BHALF,bhalf>);


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

    // (minbatch, heads, blocks, block_size, block_size)
    ctx->set_output(0, ctx->MakeShape({
      ctx->Dim(x, 0),
      ctx->MakeDim(heads),
      ctx->MakeDim(blocks),
      ctx->MakeDim(blk_size),
      ctx->MakeDim(blk_size)
    }));
    return Status::OK();
}
Status xn_shape(InferenceContext* ctx)
{
    // keys, queries, values all have same shape
    ctx->set_output(0, ctx->input(1));
    return Status::OK();
}

REGISTER_OP("BlocksparseTransformerNT")
    .Input("a: half")
    .Input("b: half")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: CT")
    .Attr("CT: {half, bfloat16}")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(nt_shape)
    .Doc(R"doc(
Multiply the dense matrix "a" by the dense matrix "b.T" and produce blocksparse output "c".
)doc");

REGISTER_OP("BlocksparseTransformerNN")
    .Input("a: half")
    .Input("b: half")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: half")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(xn_shape)
    .Doc(R"doc(
Multiply the blocksparse matrix "a" by the dense matrix "b" and produce dense output "c".
)doc");

REGISTER_OP("BlocksparseTransformerTN")
    .Input("a: half")
    .Input("b: half")
    .Input("nt_lut: int32")
    .Input("nn_lut: int32")
    .Input("tn_lut: int32")
    .Output("c: half")
    .Attr("heads: int")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("nn_max: int")
    .Attr("tn_max: int")
    .Attr("bench: int = 0")
    .SetShapeFn(xn_shape)
    .Doc(R"doc(
Multiply the blocksparse matrix "a.T" by the dense matrix "b" and produce dense output "c".
)doc");

template <typename CT, typename CV2, typename CV4> bool blocksparse_transformer_nt(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b, CT* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim);
bool blocksparse_transformer_xn(CUstream stream, const uint2* lut, const ehalf* a, const ehalf* b, ehalf* c, uint block_size, uint blocks, uint batch_dim, uint ctx_blks, uint head_dim, uint state_dim, uint lut_heads, uint lut_dim, uint op, uint magic, uint shift, uint max_blks);


#define NT_OP 0
#define NN_OP 1
#define TN_OP 2

template <typename CT, typename CV1, typename CV2, typename CV4, uint OP>
class BlocksparseTransformerOp : public OpKernel {
 public:
  explicit BlocksparseTransformerOp(OpKernelConstruction* ctx) : OpKernel(ctx), major_(0), magic_(0), shift_(0), head_state_(0)
  {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("heads",    &heads_    ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &blocks_   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size", &blk_size_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks", &ctx_blks_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("nn_max",   &nn_max_   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tn_max",   &tn_max_   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_    ));
        repeat_ = bench_ ? bench_ : 1;
        flops_ = (float)(blocks_ * blk_size_ * blk_size_);
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (major_ == 0)
    {
      GetCountSMsVersion(&major_, NULL);
      OP_REQUIRES(ctx, major_ >= 7, errors::InvalidArgument("Tensorcore GPU required"));
      if (bench_)
      {
        const char* op = OP == NT_OP ? "NT" : OP == NN_OP ? "NN" : "TN";
        sprintf(bench_string_, "op:%s bsize:%dx%d blocks:%6d ctx:%5d", op, blk_size_, blk_size_, blocks_, ctx_blks_);
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
    uint ctx_dim   = a.dim_size(1);
    uint state_dim = a.dim_size(2);

    if (head_state_ == 0)
    {
      OP_REQUIRES(ctx,
        b.dim_size(0) == batch_dim &&
        b.dim_size(1) == ctx_dim   &&
        b.dim_size(2) == state_dim, errors::InvalidArgument("Mismatched Shapes"));

      OP_REQUIRES(ctx, (state_dim & 7) == 0,                  errors::InvalidArgument("Head state dim must be multiple of 8, and ideally a multiple of 64"));
      OP_REQUIRES(ctx, ctx_dim == ctx_blks_ * blk_size_,      errors::InvalidArgument("Bad context length"));
      OP_REQUIRES(ctx, lut_heads == heads_ || lut_heads == 1, errors::InvalidArgument("Bad head dim"));
      OP_REQUIRES(ctx, state_dim % heads_ == 0,               errors::InvalidArgument("state_dim not evenly divisible by number of heads"));
      head_state_ = state_dim / heads_;
    }

    Tensor* c;
    TensorShape shapeC({batch_dim, heads_, blocks_, blk_size_, blk_size_});
    OP_REQUIRES(ctx, shapeC.num_elements() < (1ull << 32), errors::InvalidArgument("Attention space too large.  Only 32 bit address offsets currently supported (easily fixed if needed.)"));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shapeC, &c));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();
    const ehalf* a_ptr = (const ehalf*)a.flat<EHALF>().data();
    const ehalf* b_ptr = (const ehalf*)b.flat<EHALF>().data();
            CV1* c_ptr = (        CV1*)c->tensor_data().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, bench_string_, 0, flops_ * (float)(batch_dim * state_dim), repeat_);

    for (int r = 0; r < repeat_; r++)
      blocksparse_transformer_nt<CV1,CV2,CV4>(stream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_, heads_, head_state_, lut_heads, lut_dim);

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
    uint ctx_dim   = b.dim_size(1);
    uint state_dim = b.dim_size(2);

    if (head_state_ == 0)
    {
      OP_REQUIRES(ctx,
        a.dim_size(0) == batch_dim &&
        a.dim_size(1) == heads_    &&
        a.dim_size(2) == blocks_   &&
        a.dim_size(3) == blk_size_ &&
        a.dim_size(4) == blk_size_, errors::InvalidArgument("Mismatched a shape"));

      OP_REQUIRES(ctx, (state_dim & 7) == 0,                  errors::InvalidArgument("Head state dim must be multiple of 8, and ideally a multiple of 64"));
      OP_REQUIRES(ctx, ctx_dim == ctx_blks_ * blk_size_,      errors::InvalidArgument("Bad context length"));
      OP_REQUIRES(ctx, lut_heads == heads_ || lut_heads == 1, errors::InvalidArgument("Bad head dim"));
      OP_REQUIRES(ctx, state_dim % heads_ == 0,               errors::InvalidArgument("state_dim not evenly divisible by number of heads"));
      head_state_ = state_dim / heads_;

      uint div = CEIL_DIV(head_state_, 64);
      magicu64(div, magic_, shift_);
      OP_REQUIRES(ctx, magic_ > 0, errors::InvalidArgument("Bad magic for div: ", div));
    }

    Tensor* c;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b.shape(), &c));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();
    const ehalf* a_ptr = (const ehalf*)a.flat<EHALF>().data();
    const ehalf* b_ptr = (const ehalf*)b.flat<EHALF>().data();
          ehalf* c_ptr = (      ehalf*)c->tensor_data().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, bench_string_, 0, flops_ * (float)(batch_dim * state_dim), repeat_);

    for (int r = 0; r < repeat_; r++)
      blocksparse_transformer_xn(stream, l_ptr, a_ptr, b_ptr, c_ptr, blk_size_, blocks_, batch_dim, ctx_blks_, heads_, head_state_, lut_heads, lut_dim, op, magic_, shift_, max_lut);

    if (bench) delete bench;
  }
  int major_, heads_, blocks_, blk_size_, ctx_blks_, nn_max_, tn_max_, bench_, repeat_, flops_;
  uint magic_, shift_, head_state_;
  char bench_string_[256];
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerNT").Device(DEVICE_GPU).TypeConstraint<EHALF>("CT"),BlocksparseTransformerOp<EHALF,ehalf,ehalf2,ehalf4,NT_OP>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerNT").Device(DEVICE_GPU).TypeConstraint<BHALF>("CT"),BlocksparseTransformerOp<BHALF,bhalf,bhalf2,bhalf4,NT_OP>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerNN").Device(DEVICE_GPU),BlocksparseTransformerOp<EHALF,ehalf,ehalf2,ehalf4,NN_OP>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseTransformerTN").Device(DEVICE_GPU),BlocksparseTransformerOp<EHALF,ehalf,ehalf2,ehalf4,TN_OP>);


bool BlocksparseMaskedSoftmax(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const bhalf* x,
          ehalf* y,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale);

REGISTER_OP("BlocksparseMaskedSoftmax")
    .Input("x: bfloat16")
    .Input("scale: float")
    .Input("lut: int32")
    .Input("mask: MT")
    .Output("y: half")
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
    .Output("y: half")
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

    if (blk_size_ == 64)
      OP_REQUIRES(ctx, lut_max_ * blk_size_ <= 16*1024, errors::InvalidArgument("max sparse sofmax dim: 16K"));
    else
      OP_REQUIRES(ctx, lut_max_ * blk_size_ <=  8*1024, errors::InvalidArgument("max sparse sofmax dim: 8K"));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& s   = ctx->input(1);
    const Tensor& lut = ctx->input(2);
    OpInputList   mask; ctx->input_list("mask", &mask);

    OP_REQUIRES(ctx, x.dims()   == 5, errors::InvalidArgument("expecting 5 dims: (batch, head, block, blk_size, blk_size)"));
    OP_REQUIRES(ctx, lut.dims() == 3, errors::InvalidArgument("expecting 3 lut dims (head, entry, data)"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = x.dim_size(0);
    uint head_dim  = x.dim_size(1);

    uint mask_heads = 1;
    const char* m_ptr = NULL;
    if (mask.size() > 0)
    {
      OP_REQUIRES(ctx, mask[0].dims() == 3, errors::InvalidArgument("expecting 3 mask dims (head, blk_size, block)"));
      mask_heads = mask[0].dim_size(0);
      m_ptr = mask[0].tensor_data().data();
    }
    OP_REQUIRES(ctx, lut_heads  == head_dim || lut_heads  == 1, errors::InvalidArgument("Bad lut head dim"));
    OP_REQUIRES(ctx, mask_heads == head_dim || mask_heads == 1, errors::InvalidArgument("Bad mask head dim"));

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    const uint2* l_ptr = (const uint2*)lut.flat<int32>().data();
    const bhalf* x_ptr = (const bhalf*)x.flat<BHALF>().data();
          ehalf* y_ptr = (      ehalf*)y->flat<EHALF>().data();
    float scale = s.scalar<float>()();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseMaskedSoftmax(stream, l_ptr, m_ptr, x_ptr, y_ptr, blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, mask_heads, scale);
  }
  int blocks_, blk_size_, ctx_blks_, lut_max_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmax").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint8 >("MT"),BlocksparseMaskedSoftmaxOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmax").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint16>("MT"),BlocksparseMaskedSoftmaxOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmax").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint32>("MT"),BlocksparseMaskedSoftmaxOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmax").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint64>("MT"),BlocksparseMaskedSoftmaxOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseSoftmax").Device(DEVICE_GPU).HostMemory("scale"),BlocksparseMaskedSoftmaxOp);


bool BlocksparseMaskedSoftmaxGrad(CUstream stream,
    const uint2* lut,
    const  char* mask,
    const ehalf* dy,
    const ehalf* y,
          ehalf* dx,
    uint block_size, uint blocks,
    uint batch_dim,  uint head_dim, uint ctx_blks,
    uint lut_heads,  uint lut_dim,  uint max_lut,
    uint mask_heads, float scale);

REGISTER_OP("BlocksparseMaskedSoftmaxGrad")
    .Input("dy: half")
    .Input("y: half")
    .Input("scale: float")
    .Input("lut: int32")
    .Input("mask: MT")
    .Output("dx: half")
    .Attr("MT: { uint8, uint16, uint32, uint64 }")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("lut_max: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Blocksparse softmax with mask
)doc");

REGISTER_OP("BlocksparseSoftmaxGrad")
    .Input("dy: half")
    .Input("y: half")
    .Input("scale: float")
    .Input("lut: int32")
    .Output("dx: half")
    .Attr("blocks: int")
    .Attr("blk_size: int")
    .Attr("ctx_blks: int")
    .Attr("lut_max: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Blocksparse softmax without mask
)doc");

class BlocksparseMaskedSoftmaxGradOp : public OpKernel {
 public:
  explicit BlocksparseMaskedSoftmaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &blocks_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blk_size", &blk_size_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ctx_blks", &ctx_blks_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lut_max",  &lut_max_  ));

    OP_REQUIRES(ctx, lut_max_ * blk_size_ < 32768, errors::InvalidArgument("max sparse key dim: 32768 (easily upgraded if desired)"));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy  = ctx->input(0);
    const Tensor& y   = ctx->input(1);
    const Tensor& s   = ctx->input(2);
    const Tensor& lut = ctx->input(3);
    OpInputList   mask; ctx->input_list("mask", &mask);

    OP_REQUIRES(ctx, dy.dims()  == 5, errors::InvalidArgument("expecting 5 dims: (batch, head, block, blk_size, blk_size)"));
    OP_REQUIRES(ctx, lut.dims() == 3, errors::InvalidArgument("expecting 3 lut dims (head, entry, data)"));

    uint lut_heads = lut.dim_size(0);
    uint lut_dim   = lut.dim_size(1);
    uint batch_dim = dy.dim_size(0);
    uint head_dim  = dy.dim_size(1);

    uint mask_heads = 1;
    const char* m_ptr = NULL;
    if (mask.size() > 0)
    {
      OP_REQUIRES(ctx, mask[0].dims() == 3, errors::InvalidArgument("expecting 3 mask dims (head, blk_size, block)"));
      mask_heads = mask[0].dim_size(0);
      m_ptr = mask[0].tensor_data().data();
    }
    OP_REQUIRES(ctx, lut_heads  == head_dim || lut_heads  == 1, errors::InvalidArgument("Bad lut head dim"));
    OP_REQUIRES(ctx, mask_heads == head_dim || mask_heads == 1, errors::InvalidArgument("Bad mask head dim"));

    Tensor* dx = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));

    const uint2*  l_ptr = (const uint2*)lut.flat<int32>().data();
    const ehalf* dy_ptr = (const ehalf*)dy.flat<EHALF>().data();
    const ehalf*  y_ptr = (const ehalf*)y.flat<EHALF>().data();
          ehalf* dx_ptr = (      ehalf*)dx->flat<EHALF>().data();
    float scale = s.scalar<float>()();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseMaskedSoftmaxGrad(stream, l_ptr, m_ptr, dy_ptr, y_ptr, dx_ptr, blk_size_, blocks_, batch_dim, head_dim, ctx_blks_, lut_heads, lut_dim, lut_max_, mask_heads, scale);
  }
  int blocks_, blk_size_, ctx_blks_, lut_max_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint8 >("MT"),BlocksparseMaskedSoftmaxGradOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint16>("MT"),BlocksparseMaskedSoftmaxGradOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint32>("MT"),BlocksparseMaskedSoftmaxGradOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaskedSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale").TypeConstraint<uint64>("MT"),BlocksparseMaskedSoftmaxGradOp);
REGISTER_KERNEL_BUILDER(Name("BlocksparseSoftmaxGrad").Device(DEVICE_GPU).HostMemory("scale"),BlocksparseMaskedSoftmaxGradOp);
