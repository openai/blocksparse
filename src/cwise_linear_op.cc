
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

template <typename T> bool CWiseLinear_Forward(CUstream stream, T* y, const T* x, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);

REGISTER_OP("CWiseLinear")
    .Input("x: T")
    .Input("a: n_a * float")
    .Input("b: n_b * float")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("relu: bool = false")
    .Attr("swap: bool = false")
    .Attr("n_a: int >= 0")
    .Attr("n_b: int >= 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
y = a*x + b where "a" and "b" are channel vectors and x and y are in NCHW format
)doc");

template <typename T, typename V>
class CWiseLinearOp : public OpKernel {
 public:
  explicit CWiseLinearOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu", &relu_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("swap", &swap_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    OpInputList a; ctx->input_list("a", &a);
    OpInputList b; ctx->input_list("b", &b);

    uint N = x.dim_size(0);
    uint C = x.dim_size(1);

    uint rank = x.dims();
    uint DHW  = 1;
    for (uint r = 2; r < rank; r++)
      DHW *= x.dim_size(r);

    if (a.size())
    {
      OP_REQUIRES(ctx, C == a[0].shape().num_elements(), errors::InvalidArgument("CWiseLinear missmatched channels(a)"));
    }
    if (b.size())
    {
      OP_REQUIRES(ctx, C == b[0].shape().num_elements(), errors::InvalidArgument("CWiseLinear missmatched channels(b)"));
    }

    Tensor* y; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

              V* y_ptr = (V*)y->flat<T>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* a_ptr = a.size() ? a[0].flat<float>().data() : NULL;
    const float* b_ptr = b.size() ? b[0].flat<float>().data() : NULL;

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    CWiseLinear_Forward<V>(stream, y_ptr, x_ptr, a_ptr, b_ptr, N, C, DHW, relu_, swap_);
  }
  bool relu_, swap_;
};
REGISTER_KERNEL_BUILDER(Name("CWiseLinear").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),CWiseLinearOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinear").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinear").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearOp<BHALF,bhalf>);


template <typename T> bool CWiseLinear_Backward(CUstream stream, T* dx, float* da, float* db, const T* dy, const T* xy, const float* a, const float* b, uint N, uint C, uint DHW, bool relu, bool swap);

REGISTER_OP("CWiseLinearGrad")
    .Input("dy: T")
    .Input("xy: n_xy * T") // can also be y in relu(x+b) mode (we save the output rather than input for gradient pass)
    .Input("a: n_a * float")
    .Input("b: n_b * float")
    .Output("dx: T")
    .Output("da: float")
    .Output("db: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("relu: bool = false")
    .Attr("swap: bool = false")
    .Attr("n_xy: int >= 0")
    .Attr("n_a: int >= 0")
    .Attr("n_b: int >= 0")
    .SetShapeFn([](InferenceContext* ctx)
    {
      ctx->set_output(0, ctx->input(0));

      std::vector<ShapeHandle> a, b;
      ctx->input("a", &a);
      ctx->input("b", &b);

      if (a.size())
        ctx->set_output(1, a[0]);
      else
        ctx->set_output(1, ctx->UnknownShape());

      if (b.size())
        ctx->set_output(2, b[0]);
      else
        ctx->set_output(2, ctx->UnknownShape());

      return Status::OK();
    })
    .Doc(R"doc(
Gradients of y = a*x + b where "a" and "b" are channel vectors and x and y are in NCHW format
)doc");

template <typename T, typename V>
class CWiseLinearGradOp : public OpKernel {
 public:
  explicit CWiseLinearGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu", &relu_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("swap", &swap_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    OpInputList xy; ctx->input_list("xy", &xy);
    OpInputList a;  ctx->input_list("a",  &a);
    OpInputList b;  ctx->input_list("b",  &b);

    uint N = dy.dim_size(0);
    uint C = dy.dim_size(1);

    uint rank = dy.dims();
    uint DHW  = 1;
    for (uint r = 2; r < rank; r++)
      DHW *= dy.dim_size(r);

    V* dx_ptr = NULL;
    if (a.size() == 0 && !relu_)
      // no scale and no relu: just pass dy to dx
      ctx->set_output(0, dy);
    else
    {
      Tensor* dx; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));
      dx_ptr = (V*)dx->flat<T>().data();
    }

    float* da_ptr;
    if (a.size())
    {
      Tensor* da; OP_REQUIRES_OK(ctx, ctx->allocate_output(1, a[0].shape(), &da));
      da_ptr = da->flat<float>().data();
    }
    else
    {
      Tensor* da; OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape(), &da));
      da_ptr = NULL;
    }
    float* db_ptr;
    if (b.size())
    {
      Tensor* db; OP_REQUIRES_OK(ctx, ctx->allocate_output(2, b[0].shape(), &db));
      db_ptr = db->flat<float>().data();
    }
    else
    {
      Tensor* db; OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape(), &db));
      db_ptr = NULL;
    }

    const     V* dy_ptr =             (const V*)dy.flat<T>().data();
    const     V* xy_ptr = xy.size() ? (const V*)xy[0].flat<T>().data() : NULL;
    const float*  a_ptr = a.size()  ? a[0].flat<float>().data()        : NULL;
    const float*  b_ptr = b.size()  ? b[0].flat<float>().data()        : NULL;

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    CWiseLinear_Backward<V>(stream, dx_ptr, da_ptr, db_ptr, dy_ptr, xy_ptr, a_ptr, b_ptr, N, C, DHW, relu_, swap_);
  }
  bool relu_, swap_;
};
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGrad").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),CWiseLinearGradOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearGradOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearGradOp<BHALF,bhalf>);

