
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

template <typename T, typename V> bool LayerNormForward_CN(CUstream stream, int SMs, T* y, float* mean, float* rstd, float* p1, float* p2, const T* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template <typename T, typename V> bool LayerNormForward_NC(CUstream stream, int SMs, T* y, float* mean, float* rstd, const T* x, const float* g, const float* b, float epsilon, int K, int N, float rcpK, int relu);
template <typename T, typename V> bool LayerNormSegmentedForward_NC(CUstream stream, int SMs, T* y, float* mean, float* rstd, const T* x, const float* g, const float* b, float epsilon, uint N, uint S, uint K, float rcpK, int relu);

REGISTER_OP("LayerNorm")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Output("y: T")
    .Output("mean: float")
    .Output("rstd: float")
    .Output("p1: float")
    .Output("p2: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("S: int = 1")
    .Attr("axis: int")
    .Attr("epsilon: float")
    .Attr("relu: bool")
    .Attr("atomics: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {

      int S;    TF_RETURN_IF_ERROR(ctx->GetAttr("S",    &S   ));
      int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

      ShapeHandle x = ctx->input(0);

      int rank = ctx->Rank(x);
      if (rank > 0)
      {
        std::vector<DimensionHandle> dims;
        dims.push_back(ctx->MakeDim(S));
        for (int i = 0; i < rank; i++)
          if (i != axis)
            dims.push_back(ctx->Dim(x, i));
        ShapeHandle n = ctx->MakeShape(dims);

        ctx->set_output(0, x);
        ctx->set_output(1, n);
        ctx->set_output(2, n);
      }
      else
      {
        ctx->set_output(0, ctx->UnknownShape());
        ctx->set_output(1, ctx->UnknownShape());
        ctx->set_output(2, ctx->UnknownShape());
      }
      ctx->set_output(3, ctx->UnknownShape());
      ctx->set_output(4, ctx->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Layer norm applied to blocks full layer
)doc");

template <typename T, typename V1, typename V4>
class LayerNormOp : public OpKernel {
 public:
  explicit LayerNormOp(OpKernelConstruction* ctx) : OpKernel(ctx), K_(0), rcpK_(0.0f), SMs_(0) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("S",       &S_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu",    &relu_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",    &axis_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("atomics", &atomics_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",   &bench_  ));
    repeat_ = bench_ ? bench_ : 1;
  }

  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& x = ctx->input(0);
    const Tensor& g = ctx->input(1);
    const Tensor& b = ctx->input(2);

    if (axis_ < 0)
      axis_ += x.dims();

    int N = 1, K = x.dim_size(axis_), last_dim = x.dims()-1;
    TensorShape shapeN;
    shapeN.AddDim(S_);
    for (int i = 0; i <= last_dim; i++)
    {
      if (i != axis_)
      {
        shapeN.AddDim(x.dim_size(i));
        N *= x.dim_size(i);
      }
    }
    OP_REQUIRES(ctx, axis_ != 0 || (N & 3) == 0, errors::InvalidArgument("Sum of non-feature axis dims needs to be multiple of 4 for feature axis=0."));

    if (K_ == 0)
    {
      OP_REQUIRES(ctx, K == g.shape().num_elements(), errors::InvalidArgument("Bad Gain Shape"));
      OP_REQUIRES(ctx, K == b.shape().num_elements(), errors::InvalidArgument("Bad Bias Shape"));
      OP_REQUIRES(ctx, (K % S_) == 0, errors::InvalidArgument("Shape not evenly devided by segments"));
      OP_REQUIRES(ctx, axis_ != 0 || S_ == 1, errors::InvalidArgument("Segments only implemented on axis=1 for now"));
      K_    = K / S_;
      rcpK_ = 1.0f / (float)K_;
      SMs_  = GetCountSMs();
    }

    Tensor*    y = nullptr;
    Tensor* mean = nullptr;
    Tensor* rstd = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(),    &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1,    shapeN, &mean));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2,    shapeN, &rstd));

    float* p1_ptr = nullptr;
    float* p2_ptr = nullptr;
    Tensor* P1 = nullptr;
    Tensor* P2 = nullptr;
    TensorShape shapeP;
    if (axis_ == 0)
    {
      shapeP.AddDim(SMs_*2);
      shapeP.AddDim(N);
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3,  shapeP, &P1));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4,  shapeP, &P2));
    if (axis_ == 0)
    {
      p1_ptr = P1->flat<float>().data();
      p2_ptr = P2->flat<float>().data();
    }
             V1*    y_ptr = (V1*)y->flat<T>().data();
          float* mean_ptr = mean->flat<float>().data();
          float* rstd_ptr = rstd->flat<float>().data();
    const    V1*    x_ptr = (const V1*)x.flat<T>().data();
    const float*    g_ptr = g.flat<float>().data();
    const float*    b_ptr = b.flat<float>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, "LayerNormForward", N*S_*K_*2*sizeof(T), 0, repeat_);

    for (int r = 0; r < repeat_; r++)
      if (axis_ == last_dim)
      {
        if (S_ > 1 || K_ <= 1024*8)
          LayerNormSegmentedForward_NC<V1,V4>(stream, SMs_, y_ptr, mean_ptr, rstd_ptr, x_ptr, g_ptr, b_ptr, epsilon_, N, S_, K_, rcpK_, relu_);
        else
          LayerNormForward_NC<V1,V4>(stream, SMs_, y_ptr, mean_ptr, rstd_ptr, x_ptr, g_ptr, b_ptr, epsilon_, K_, N, rcpK_, relu_);
      }
      else
        LayerNormForward_CN<V1,V4>(stream, SMs_, y_ptr, mean_ptr, rstd_ptr, p1_ptr, p2_ptr, x_ptr, g_ptr, b_ptr, epsilon_, K_, N, rcpK_, relu_);

    if (bench) delete bench;
  }
  float epsilon_, rcpK_;
  int S_, K_, axis_, SMs_, bench_, repeat_;
  bool relu_, atomics_;
};
REGISTER_KERNEL_BUILDER(Name("LayerNorm").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), LayerNormOp<FLOAT,float,float4>);
REGISTER_KERNEL_BUILDER(Name("LayerNorm").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), LayerNormOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LayerNorm").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), LayerNormOp<BHALF,bhalf,bhalf4>);

template <typename T, typename V> bool LayerNormBackward_NC(CUstream stream, int SMs, T* dx, float* dg, float* db, const T* dy, const T* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template <typename T, typename V> bool LayerNormBackward_CN(CUstream stream, int SMs, T* dx, float* dg, float* db, float* sum1, float* sum2, const T* dy, const T* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, int K, int N, float rcpK, int relu);
template <typename T, typename V> bool LayerNormSegmentedBackward_NC(CUstream stream, int SMs, T* dx, float* dg, float* db, const T* dy, const T* x, const float* g, const float* b, const float* mean, const float* rstd, float epsilon, uint N, uint S, uint K, float rcpK, int relu, int atomics);

REGISTER_OP("LayerNormGrad")
    .Input("dy: T")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Input("mean: float")
    .Input("rstd: float")
    .Output("dx: T")
    .Output("dg: float")
    .Output("db: float")
    .Output("p1: float")
    .Output("p2: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("S: int = 1")
    .Attr("axis: int")
    .Attr("epsilon: float")
    .Attr("relu: bool")
    .Attr("atomics: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      ctx->set_output(2, ctx->input(3));
      ctx->set_output(3, ctx->UnknownShape());
      ctx->set_output(4, ctx->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
Gradients of layer norm applied to full layer
)doc");

template <typename T, typename V1, typename V4>
class LayerNormGradOp : public OpKernel {
 public:
  explicit LayerNormGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), K_(0), rcpK_(0.0f), SMs_(0) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("S",       &S_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu",    &relu_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",    &axis_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("atomics", &atomics_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",   &bench_  ));
    repeat_ = bench_ ? bench_ : 1;
  }
  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& dy = ctx->input(0);
    const Tensor& x  = ctx->input(1);
    const Tensor& g  = ctx->input(2);
    const Tensor& b  = ctx->input(3);
    const Tensor& mean = ctx->input(4);
    const Tensor& rstd = ctx->input(5);

    if (axis_ < 0)
      axis_ += x.dims();

    int rank = x.dims();
    int N = 1, K = x.dim_size(axis_);
    for (int i = 0; i < rank; i++)
      if (i != axis_)
        N *= x.dim_size(i);

    if (K_ == 0)
    {
      K_    = K / S_;
      rcpK_ = 1.0f / (float)K_;
      SMs_  = GetCountSMs();
    }

    Tensor* dx = nullptr;
    Tensor* dg = nullptr;
    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &dg));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, b.shape(), &db));

    Tensor* P1 = nullptr;
    Tensor* P2 = nullptr;
    TensorShape shapeP;
    if (axis_ == 0)
    {
      shapeP.AddDim(SMs_*2);
      shapeP.AddDim(N);
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3,  shapeP, &P1));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4,  shapeP, &P2));
    float* p1_ptr = nullptr;
    float* p2_ptr = nullptr;
    if (axis_ == 0)
    {
      p1_ptr = P1->flat<float>().data();
      p2_ptr = P2->flat<float>().data();
    }

             V1*   dx_ptr = (V1*)dx->flat<T>().data();
          float*   dg_ptr = dg->flat<float>().data();
          float*   db_ptr = db->flat<float>().data();
    const    V1*   dy_ptr = (const V1*)dy.flat<T>().data();
    const    V1*    x_ptr = (const V1*)x.flat<T>().data();
    const float*    g_ptr = g.flat<float>().data();
    const float*    b_ptr = b.flat<float>().data();
    const float* mean_ptr = mean.flat<float>().data();
    const float* rstd_ptr = rstd.flat<float>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_) bench = new Benchmark(stream, "LayerNormBackward", N*S_*K_*5*sizeof(T), 0, repeat_);

    for (int r = 0; r < repeat_; r++)
      if (axis_ == 0)
        LayerNormBackward_CN<V1,V4>(stream, SMs_, dx_ptr, dg_ptr, db_ptr, p1_ptr, p2_ptr, dy_ptr, x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, epsilon_, K_, N, rcpK_, relu_);
      else
      {
        if (S_ > 1 || K_ <= 1024*8)
          LayerNormSegmentedBackward_NC<V1,V4>(stream, SMs_, dx_ptr, dg_ptr, db_ptr, dy_ptr, x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, epsilon_, N, S_, K_, rcpK_, relu_, atomics_);
        else
          LayerNormBackward_NC<V1,V4>(stream, SMs_, dx_ptr, dg_ptr, db_ptr, dy_ptr, x_ptr, g_ptr, b_ptr, mean_ptr, rstd_ptr, epsilon_, K_, N, rcpK_, relu_);
      }

    if (bench) delete bench;
  }
  float epsilon_, rcpK_;
  int S_, K_, axis_, SMs_, bench_, repeat_;
  bool relu_, atomics_;
};
REGISTER_KERNEL_BUILDER(Name("LayerNormGrad").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), LayerNormGradOp<FLOAT,float,float4>);
REGISTER_KERNEL_BUILDER(Name("LayerNormGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), LayerNormGradOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LayerNormGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), LayerNormGradOp<BHALF,bhalf,bhalf4>);


#define OP_GAT 0
#define OP_SCT 1
#define OP_ADD 2
#define OP_MUL 3

template <typename T, typename V4, typename V8>
bool SparseOp(CUstream stream,
            T* z,
    const   T* x,
    const   T* y,
    const int* lut,
    int op, int K, int N);

template <typename T, typename V4, typename V8>
bool SparseMulGrad(CUstream stream,
            T* dx,
            T* dy,
    const   T* dz,
    const   T* x,
    const   T* y,
    const int* lut,
    int K, int N);

REGISTER_OP("GatherScatter")
    .Input("x: T")
    .Input("gather: int32")
    .Input("scatter: int32")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("K: int")
    .Attr("op: int")
    .SetShapeFn([](InferenceContext* ctx)
    {
      int K; TF_RETURN_IF_ERROR(ctx->GetAttr("K", &K));

      ShapeHandle x = ctx->input(0);
      int rank = ctx->Rank(x);
      if (rank > 0)
      {
        std::vector<DimensionHandle> shape;
        shape.reserve(rank);
        for (int i = 0; i < rank; i++)
            shape.push_back(i == 0 ? ctx->MakeDim(K) : ctx->Dim(x, i));
        ctx->set_output(0, ctx->MakeShape(shape));
      }
      return Status::OK();
    })
    .Doc(R"doc(
Take a sparse feature slice out of a tensor
)doc");

template <typename T, typename V1, typename V4, typename V8>
class GatherScatterOp : public OpKernel {
 public:
  explicit GatherScatterOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("C",  &C_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",  &K_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& x   = ctx->input(0);
    const Tensor& lut = ctx->input(1);

    int N = 1;
    int rank = x.dims();
    TensorShape shape({K_});
    for (int i = 1; i < rank; i++) {
      shape.AddDim(x.dim_size(i));
      N *= x.dim_size(i);
    }
    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &y));

           V1* y_ptr = (V1*)y->flat<T>().data();
    const  V1* x_ptr = (const V1*)x.flat<T>().data();
    const int* l_ptr = lut.flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    SparseOp<V1,V4,V8>(stream, y_ptr, x_ptr, 0, l_ptr, op_, K_, N);
  }
  int C_, K_, op_;
};
REGISTER_KERNEL_BUILDER(Name("GatherScatter").Device(DEVICE_GPU).TypeConstraint<float>("T"), GatherScatterOp<float,float,float4,float8>);
REGISTER_KERNEL_BUILDER(Name("GatherScatter").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), GatherScatterOp<EHALF,ehalf,ehalf4,ehalf8>);
REGISTER_KERNEL_BUILDER(Name("GatherScatter").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), GatherScatterOp<BHALF,bhalf,bhalf4,bhalf8>);


REGISTER_OP("ScatterAddMul")
    .Input("x: T")
    .Input("y: T")
    .Input("gather: int32")
    .Input("scatter: int32")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("K: int")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Take smaller tensor (y) and sparsely add or multiply with larger tensor (x).
)doc");


template <typename T, typename V1, typename V4, typename V8>
class ScatterAddMulOp : public OpKernel {
 public:
  explicit ScatterAddMulOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("C",  &C_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",  &K_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);
    const Tensor& gather  = ctx->input(2);
    const Tensor& scatter = ctx->input(3);

    int N = 1;
    int rank = x.dims();
    for (int i = 1; i < rank; i++)
      N *= x.dim_size(i);

    int K;
    V1* z_ptr;
    const int* l_ptr;
    if (op_ == OP_ADD)
    {
      K = C_;
      ctx->set_output(0, x); // gradient doesn't need to remember x so just add on top of x
      z_ptr = (V1*)x.flat<T>().data();
      l_ptr = gather.flat<int32>().data(); // only need to touch indexes in smaller gather lut
    }
    else
    {
      K = K_;
      Tensor* z = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));
      z_ptr = (V1*)z->flat<T>().data();
      l_ptr = scatter.flat<int32>().data();
    }
    const V1* x_ptr = (const V1*)x.flat<T>().data();
    const V1* y_ptr = (const V1*)y.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    SparseOp<V1,V4,V8>(stream, z_ptr, x_ptr, y_ptr, l_ptr, op_, K, N);
  }
  int C_, K_, op_;
};
REGISTER_KERNEL_BUILDER(Name("ScatterAddMul").Device(DEVICE_GPU).TypeConstraint<float>("T"), ScatterAddMulOp<float,float,float4,float8>);
REGISTER_KERNEL_BUILDER(Name("ScatterAddMul").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), ScatterAddMulOp<EHALF,ehalf,ehalf4,ehalf8>);
REGISTER_KERNEL_BUILDER(Name("ScatterAddMul").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), ScatterAddMulOp<BHALF,bhalf,bhalf4,bhalf8>);

REGISTER_OP("ScatterMulGrad")
    .Input("dz: T")
    .Input("x: T")
    .Input("y: T")
    .Input("gather: int32")
    .Output("dx: T")
    .Output("dy: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("K: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Grad of SparseMul
)doc");

template <typename T, typename V1, typename V4, typename V8>
class ScatterMulGradOp : public OpKernel {
 public:
  explicit ScatterMulGradOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("C",  &C_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",  &K_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& dz   = ctx->input(0);
    const Tensor& x    = ctx->input(1);
    const Tensor& y    = ctx->input(2);
    const Tensor& lut  = ctx->input(3);

    int N = 1;
    int rank = x.dims();
    for (int i = 1; i < rank; i++)
      N *= x.dim_size(i);

    Tensor* dy = nullptr;
    ctx->set_output(0, dz); // write dx on top of dz
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, y.shape(), &dy));

           V1* dx_ptr = (V1*)dz.flat<T>().data(); // write dx on top of dz
           V1* dy_ptr = (V1*)dy->flat<T>().data();
    const  V1* dz_ptr = (const V1*)dx_ptr;
    const  V1*  x_ptr = (const V1*)x.flat<T>().data();
    const  V1*  y_ptr = (const V1*)y.flat<T>().data();
    const int*  l_ptr = lut.flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    SparseMulGrad<V1,V4,V8>(stream, dx_ptr, dy_ptr, dz_ptr, x_ptr, y_ptr, l_ptr, C_, N);
  }
  int C_, K_;
};
REGISTER_KERNEL_BUILDER(Name("ScatterMulGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), ScatterMulGradOp<float,float,float4,float8>);
REGISTER_KERNEL_BUILDER(Name("ScatterMulGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), ScatterMulGradOp<EHALF,ehalf,ehalf4,ehalf8>);
REGISTER_KERNEL_BUILDER(Name("ScatterMulGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), ScatterMulGradOp<BHALF,bhalf,bhalf4,bhalf8>);
