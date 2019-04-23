
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
    .SetShapeFn([](InferenceContext* ctx)
    {
      ShapeHandle x = ctx->input(0);
      if (ctx->RankKnown(x))
      {
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
      }
      else
      {
        ctx->set_output(0, ctx->UnknownShape());
        ctx->set_output(1, ctx->UnknownShape());
      }
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

template <typename T, typename V4> bool Transpose_2D(CUstream stream, T* y, const T* x, uint D0, uint D1);

REGISTER_OP("Transpose2D")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &x));
      ctx->set_output(0, ctx->MakeShape({ ctx->Dim(x,1), ctx->Dim(x,0) }));
      return Status::OK();
    })
    .Doc(R"doc(
Simple/fast 2D Transpose
)doc");

template <typename T, typename V1, typename V4>
class Transpose2DOp : public OpKernel {
 public:
  explicit Transpose2DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    OP_REQUIRES(ctx, x.dims() == 2, errors::Internal("x.dims() == 2: ", x.dims()));

    uint D0 = x.dim_size(0);
    uint D1 = x.dim_size(1);

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({D1, D0}), &y));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Transpose_2D<V1,V4>(stream, (V1*)y->flat<T>().data(), (const V1*)x.flat<T>().data(), D0, D1);
  }
};
REGISTER_KERNEL_BUILDER(Name("Transpose2D").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),Transpose2DOp<FLOAT,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Transpose2D").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),Transpose2DOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Transpose2D").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),Transpose2DOp<BHALF,bhalf,bhalf4>);


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


template <typename TL> bool SoftmaxCrossEntropy(CUstream stream, ehalf* grad, float* loss, const ehalf* logits, const TL* labels, uint N, uint K);

REGISTER_OP("SoftmaxCrossEntropy")
    .Input("logits: half")
    .Input("labels: TL")
    .Output("loss: float")
    .Output("grad: half")
    .Attr("TL: { uint8, uint16, int32 }")
    .SetShapeFn([](InferenceContext* ctx) {

      ShapeHandle logits = ctx->input(0);

      int rank = ctx->Rank(logits) - 1;
      if (rank > 0)
      {
        std::vector<DimensionHandle> dims;
        dims.reserve(rank);
        for (int i = 0; i < rank; ++i)
            dims.emplace_back(ctx->Dim(logits, i));

        ctx->set_output(0, ctx->MakeShape(dims));
        ctx->set_output(1, logits);
      }
      else
      {
        ctx->set_output(0, ctx->UnknownShape());
        ctx->set_output(1, ctx->UnknownShape());
      }

      return Status::OK();
    })
    .Doc(R"doc(
SoftmaxCrossEntropy
)doc");

template <typename TL>
class SoftmaxCrossEntropyOp : public OpKernel
{
 public:
  explicit SoftmaxCrossEntropyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override
  {

    const Tensor& logits = ctx->input(0);
    const Tensor& labels = ctx->input(1);

    uint rank = logits.dims() - 1;

    uint K = logits.dim_size(rank);
    uint N  = 1;
    TensorShape l_shape;

    for (uint i = 0; i < rank; i++)
    {
      N *= logits.dim_size(i);
      l_shape.AddDim(logits.dim_size(i));
    }

    OP_REQUIRES(ctx, N == labels.shape().num_elements(), errors::InvalidArgument("Bad labels shape"));
    OP_REQUIRES(ctx, (K & 7) == 0 || (K < 256 && (K & 1) == 0), errors::InvalidArgument("Feature dim needs to be multiple of 8 or multiple of 2 if less than 256"));
    OP_REQUIRES(ctx, K <= 65536,   errors::InvalidArgument("Feature dim needs to be less than 64k"));

    Tensor* loss = nullptr;
    Tensor* grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, l_shape, &loss));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, logits.shape(), &grad));


    const ehalf* logits_ptr = (const ehalf*)logits.flat<EHALF>().data();
    const    TL* labels_ptr = (const    TL*)labels.flat<TL>().data();
          float*   loss_ptr = (float*)loss->flat<float>().data();
          ehalf*   grad_ptr = (ehalf*)grad->flat<EHALF>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    SoftmaxCrossEntropy<TL>(stream, grad_ptr, loss_ptr, logits_ptr, labels_ptr, N, K);
  }
};

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropy").Device(DEVICE_GPU).TypeConstraint<uint8 >("TL"),SoftmaxCrossEntropyOp<uint8 >);
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropy").Device(DEVICE_GPU).TypeConstraint<uint16>("TL"),SoftmaxCrossEntropyOp<uint16>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropy").Device(DEVICE_GPU).TypeConstraint< int32>("TL"),SoftmaxCrossEntropyOp< int32>);



bool SoftmaxCrossEntropyGrad(CUstream stream, uint SMs, ehalf* dx, const float* dy, const ehalf* y, uint NK, uint K);

REGISTER_OP("SoftmaxCrossEntropyGrad")
    .Input("y: half")
    .Input("dy: float")
    .Output("dx: half")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
SoftmaxCrossEntropyGrad
)doc");

class SoftmaxCrossEntropyGradOp : public OpKernel
{
 public:
  explicit SoftmaxCrossEntropyGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {}
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor&  y = ctx->input(0);
    const Tensor& dy = ctx->input(1);

    uint K  = y.dim_size(y.dims() - 1);
    uint NK = y.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &dx));

    const ehalf*  y_ptr = (const ehalf*)y.flat<EHALF>().data();
    const float* dy_ptr = (const float*)dy.flat<float>().data();
          ehalf* dx_ptr = (ehalf*)dx->flat<EHALF>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    SoftmaxCrossEntropyGrad(stream, SMs_, dx_ptr, dy_ptr, y_ptr, NK, K);
  }
  uint SMs_;
};

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyGrad").Device(DEVICE_GPU),SoftmaxCrossEntropyGradOp);