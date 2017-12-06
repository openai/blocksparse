
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/platform/stream_executor.h"
// #include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "gpu_types.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// using perftools::gputools::cuda::AsCUDAStreamValue;

template <typename T> bool CWiseLinearAXPB_Forward(CUstream stream, T* y, const T* x, const float* a, const float* b, int N, int C, int DHW);
template <typename T> bool CWiseLinearXPB_Backward(CUstream stream, float* db, const T* dy, int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);
template <typename TX, typename TY> bool CWiseLinearAXPB_Backward(CUstream stream, TY* dx, float* da, float* db, const TY* dy, const TX* x, const float* a, int C, int NDHW, int DHW, int magic_DHW, int shift_DHW);

Status UnchangedShape(shape_inference::InferenceContext* ctx) {
  ctx->set_output(0, ctx->input(0));
  return Status::OK();
}

REGISTER_OP("CWiseLinearAXPB")
    .Input("x: T")
    .Input("a: float")
    .Input("b: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
y = a*x + b where "a" and "b" are channel vectors and x and y are in NCHW format
)doc");

REGISTER_OP("CWiseLinearAX")
    .Input("x: T")
    .Input("a: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
y = a*x where "a" is a channel vector and x and y are in NCHW format
)doc");

REGISTER_OP("CWiseLinearXPB")
    .Input("x: T")
    .Input("b: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
y = x + b where "b" is a channel vector and x and y are in NCHW format
)doc");

REGISTER_OP("CWiseLinearGradAXPB")
    .Input("dy: TY")
    .Input("x: TX")
    .Input("a: float")
    .Input("b: float")
    .Output("dx: TY")
    .Output("da: float")
    .Output("db: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .Attr("magic_DHW: int")
    .Attr("shift_DHW: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      ctx->set_output(2, ctx->input(3));
      return Status::OK();
    })
    .Doc(R"doc(
Gradients of y = a*x + b where "a" and "b" are channel vectors and x and y are in NCHW format
)doc");

REGISTER_OP("CWiseLinearGradAX")
    .Input("dy: TY")
    .Input("x: TX")
    .Input("a: float")
    .Output("dx: TY")
    .Output("da: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .Attr("magic_DHW: int")
    .Attr("shift_DHW: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Gradients of y = a*x where "a" is a channel vector and x and y are in NCHW format
)doc");

REGISTER_OP("CWiseLinearGradXPB")
    .Input("dy: T")
    .Input("b: float")
    .Output("db: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("C: int")
    .Attr("DHW: int")
    .Attr("magic_DHW: int")
    .Attr("shift_DHW: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
Gradients of y = x + b where "b" is a channel vector and x and y are in NCHW format
)doc");



template <typename T, typename V>
class CWiseLinearAXPBOp : public OpKernel {
 public:
  explicit CWiseLinearAXPBOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(  "C",   &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& a = ctx->input(1);
    const Tensor& b = ctx->input(2);

    int N = x.dim_size(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

              V* y_ptr = (V*)y->flat<T>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* a_ptr = a.flat<float>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearAXPB_Forward<V>(stream, y_ptr, x_ptr, a_ptr, b_ptr, N, C_, DHW_);
  }
  int C_, DHW_;
};
template <typename T, typename V>
class CWiseLinearAXOp : public OpKernel {
 public:
  explicit CWiseLinearAXOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(  "C",   &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& a = ctx->input(1);

    int N = x.dim_size(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

              V* y_ptr = (V*)y->flat<T>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* a_ptr = a.flat<float>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearAXPB_Forward<V>(stream, y_ptr, x_ptr, a_ptr, 0, N, C_, DHW_);
  }
  int C_, DHW_;
};
template <typename T, typename V>
class CWiseLinearXPBOp : public OpKernel {
 public:
  explicit CWiseLinearXPBOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(  "C",   &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& b = ctx->input(1);

    int N = x.dim_size(0);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

              V* y_ptr = (V*)y->flat<T>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearAXPB_Forward<V>(stream, y_ptr, x_ptr, 0, b_ptr, N, C_, DHW_);
  }
  int C_, DHW_;
};


template <typename TX, typename TY, typename VX, typename VY>
class CWiseLinearGradAXPBOp : public OpKernel {
 public:
  explicit CWiseLinearGradAXPBOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr(        "C",         &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "DHW",       &DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_DHW", &magic_DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_DHW", &shift_DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor& x  = ctx->input(1);
    const Tensor& a  = ctx->input(2);
    const Tensor& b  = ctx->input(3);

    int N = x.dim_size(0);

    Tensor* dx = nullptr;
    Tensor* da = nullptr;
    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, a.shape(), &da));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, b.shape(), &db));

             VY* dx_ptr = (VY*)dx->flat<TY>().data();
          float* da_ptr = da->flat<float>().data();
          float* db_ptr = db->flat<float>().data();
    const    VY* dy_ptr = (const VY*)dy.flat<TY>().data();
    const    VX*  x_ptr = (const VX*)x.flat<TX>().data();
    const float*  a_ptr = a.flat<float>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearAXPB_Backward<VX,VY>(stream, dx_ptr, da_ptr, db_ptr, dy_ptr, x_ptr, a_ptr, C_, N*DHW_, DHW_, magic_DHW_, shift_DHW_);
  }
  int C_, DHW_, magic_DHW_, shift_DHW_;
};
template <typename TX, typename TY, typename VX, typename VY>
class CWiseLinearGradAXOp : public OpKernel {
 public:
  explicit CWiseLinearGradAXOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr(        "C",         &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "DHW",       &DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_DHW", &magic_DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_DHW", &shift_DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor& x  = ctx->input(1);
    const Tensor& a  = ctx->input(2);

    int N = x.dim_size(0);

    Tensor* dx = nullptr;
    Tensor* da = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, a.shape(), &da));

             VY* dx_ptr = (VY*)dx->flat<TY>().data();
          float* da_ptr = da->flat<float>().data();
    const    VY* dy_ptr = (const VY*)dy.flat<TY>().data();
    const    VX*  x_ptr = (const VX*)x.flat<TX>().data();
    const float*  a_ptr = a.flat<float>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearAXPB_Backward<VX,VY>(stream, dx_ptr, da_ptr, 0, dy_ptr, x_ptr, a_ptr, C_, N*DHW_, DHW_, magic_DHW_, shift_DHW_);
  }
  int C_, DHW_, magic_DHW_, shift_DHW_;
};
template <typename T, typename V>
class CWiseLinearGradXPBOp : public OpKernel {
 public:
  explicit CWiseLinearGradXPBOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr(        "C",         &C_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "DHW",       &DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_DHW", &magic_DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_DHW", &shift_DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor& b  = ctx->input(1);

    int N = dy.dim_size(0);

    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b.shape(), &db));

          float* db_ptr = db->flat<float>().data();
    const     V* dy_ptr = (const V*)dy.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    CWiseLinearXPB_Backward<V>(stream, db_ptr, dy_ptr, C_, N*DHW_, DHW_, magic_DHW_, shift_DHW_);
  }
  int C_, DHW_, magic_DHW_, shift_DHW_;
};


REGISTER_KERNEL_BUILDER(Name("CWiseLinearAXPB").Device(DEVICE_GPU).TypeConstraint<float>("T"),CWiseLinearAXPBOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearAXPB").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearAXPBOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearAXPB").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearAXPBOp<BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("CWiseLinearAX"  ).Device(DEVICE_GPU).TypeConstraint<float>("T"),CWiseLinearAXOp  <float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearAX"  ).Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearAXOp  <EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearAX"  ).Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearAXOp  <BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("CWiseLinearXPB" ).Device(DEVICE_GPU).TypeConstraint<float>("T"),CWiseLinearXPBOp <float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearXPB" ).Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearXPBOp <EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearXPB" ).Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearXPBOp <BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAXPB").Device(DEVICE_GPU).TypeConstraint<float>("TX").TypeConstraint<float>("TY"),CWiseLinearGradAXPBOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAXPB").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<EHALF>("TY"),CWiseLinearGradAXPBOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAXPB").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<float>("TY"),CWiseLinearGradAXPBOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAXPB").Device(DEVICE_GPU).TypeConstraint<BHALF>("TX").TypeConstraint<BHALF>("TY"),CWiseLinearGradAXPBOp<BHALF,BHALF,bhalf,bhalf>);

REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAX").Device(DEVICE_GPU).TypeConstraint<float>("TX").TypeConstraint<float>("TY"),CWiseLinearGradAXOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAX").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<EHALF>("TY"),CWiseLinearGradAXOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAX").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<float>("TY"),CWiseLinearGradAXOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradAX").Device(DEVICE_GPU).TypeConstraint<BHALF>("TX").TypeConstraint<BHALF>("TY"),CWiseLinearGradAXOp<BHALF,BHALF,bhalf,bhalf>);

REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradXPB").Device(DEVICE_GPU).TypeConstraint<float>("T"),CWiseLinearGradXPBOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradXPB").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),CWiseLinearGradXPBOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("CWiseLinearGradXPB").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),CWiseLinearGradXPBOp<BHALF,bhalf>);


//REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU),FloatCastOp);