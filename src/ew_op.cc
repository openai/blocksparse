
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


Status UnchangedShape(shape_inference::InferenceContext* ctx) {
  ctx->set_output(0, ctx->input(0));
  return Status::OK();
}

template <typename T, typename V> bool EW_Forward(CUstream stream, T* z, const T* x, const T* y, const float* b, float alpha, int size, int N, int op);
template <typename B, typename F, typename VB, typename VF> bool EW_Backward(CUstream stream, B* dx, B* dy, float* db, const B* dz, const F* x, const F* y, const F* z, const float* g, float alpha, int size, int N, int op);

REGISTER_OP("EwZXy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
binary elementwise op
)doc");

template <typename T, typename V1, typename V4>
class EwZXyOp : public OpKernel {
 public:
  explicit EwZXyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);

    int size = x.shape().num_elements();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

          V1* z_ptr = (V1*)z->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();
    const V1* y_ptr = (const V1*)y.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, y_ptr, 0, 1.0f, size, 0, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXyOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXyOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXyOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwZXa")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise op (with optional alpha)
)doc");

template <typename T, typename V1, typename V4>
class EwZXaOp : public OpKernel {
 public:
  explicit EwZXaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int size = x.shape().num_elements();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

          V1* z_ptr = (V1*)z->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, 0, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXaOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXaOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXaOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwZXb")
    .Input("x: T")
    .Input("b: float")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
broadcast elementwise op
)doc");

template <typename T, typename V1, typename V4>
class EwZXbOp : public OpKernel {
 public:
  explicit EwZXbOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& b = ctx->input(1);

    int rank = x.dims();
    int K    = x.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

             V1* z_ptr = (V1*)z->flat<T>().data();
    const    V1* x_ptr = (const V1*)x.flat<T>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, 0, b_ptr, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXbOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXbOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXbOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("EwDxdyDzxy")
    .Input("dz: B")
    .Input("x: F")
    .Input("y: F")
    .Output("dx: B")
    .Output("dy: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
binary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxdyDzxyOp : public OpKernel {
 public:
  explicit EwDxdyDzxyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);
    const Tensor&  y = ctx->input(2);

    int size = x.shape().num_elements();

    Tensor* dx = nullptr;
    Tensor* dy = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, y.shape(), &dy));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
          VB1* dy_ptr = (VB1*)dy->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  x_ptr = (const VF1*)x.flat<F>().data();
    const VF1*  y_ptr = (const VF1*)y.flat<F>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, dy_ptr, 0, dz_ptr, x_ptr, y_ptr, 0, 0, 1.0f, size, 0, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxdyDzxyOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxdyDzxyOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxdyDzxyOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxdyDzxyOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxdyDzxyOp<float,BHALF,float,bhalf,float4,bhalf4>);

REGISTER_OP("EwDxDzza")
    .Input("dz: B")
    .Input("z: F")
    .Output("dx: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxDzzaOp : public OpKernel {
 public:
  explicit EwDxDzzaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  z = ctx->input(1);

    int size = z.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, z.shape(), &dx));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  z_ptr = (const VF1*)z.flat<F>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, 0, dz_ptr, 0, 0, z_ptr, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxDzzaOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxDzzaOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxDzzaOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxDzzaOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxDzzaOp<float,BHALF,float,bhalf,float4,bhalf4>);


REGISTER_OP("EwDxDzxa")
    .Input("dz: B")
    .Input("x: F")
    .Output("dx: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxDzxaOp : public OpKernel {
 public:
  explicit EwDxDzxaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);

    int size = x.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  x_ptr = (const VF1*)x.flat<F>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, 0, dz_ptr, x_ptr, 0, 0, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxDzxaOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxDzxaOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxDzxaOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxDzxaOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxDzxaOp<float,BHALF,float,bhalf,float4,bhalf4>);


REGISTER_OP("EwDbDzb")
    .Input("dz: B")
    .Input("b: float")
    .Output("db: float")
    .Attr("B: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename VB1, typename VB4>
class EwDbDzbOp : public OpKernel {
 public:
  explicit EwDbDzbOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  b = ctx->input(1);

    int rank = dz.dims();
    int K    = dz.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= dz.dim_size(--rank);

    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b.shape(), &db));

          float* db_ptr = db->flat<float>().data();
    const   VB1* dz_ptr = (const VB1*)dz.flat<B>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Backward<VB1,VB1,VB4,VB4>(stream, 0, 0, db_ptr, dz_ptr, 0, 0, 0, 0, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<float>("B"),EwDbDzbOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<EHALF>("B"),EwDbDzbOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<BHALF>("B"),EwDbDzbOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwDxdgDzxg")
    .Input("dz: B")
    .Input("x: F")
    .Input("g: float")
    .Output("dx: B")
    .Output("dg: float")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxdgDzxgOp : public OpKernel {
 public:
  explicit EwDxdgDzxgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);
    const Tensor&  g = ctx->input(2);

    int rank = x.dims();
    int K    = x.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    Tensor* dx = nullptr;
    Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &dg));

            VB1* dx_ptr = (VB1*)dx->flat<B>().data();
          float* dg_ptr = dg->flat<float>().data();
    const   VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const   VF1*  x_ptr = (const VF1*)x.flat<F>().data();
    const float*  g_ptr = g.flat<float>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, dg_ptr, dz_ptr, x_ptr, 0, 0, g_ptr, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxdgDzxgOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxdgDzxgOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxdgDzxgOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxdgDzxgOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
//REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxdgDzxgOp<float,BHALF,float,bhalf,float4,bhalf4>);

template <typename TY, typename TX, typename VY, typename VX> bool FloatCast(CUstream stream, TY* y, const TX* x, int size);

REGISTER_OP("FloatCast")
    .Input("x: TX")
    .Output("y: TY")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("dx_dtype: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
float to half and half to float
)doc");

template <typename TY, typename TX, typename VY1, typename VX1, typename VY4, typename VX4>
class FloatCastOp : public OpKernel {
 public:
  explicit FloatCastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int size = x.shape().num_elements();

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

          VY1* y_ptr = (VY1*)y->flat<TY>().data();
    const VX1* x_ptr = (const VX1*)x.flat<TX>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    FloatCast<VY1,VX1,VY4,VX4>(stream, y_ptr, x_ptr, size);
  }
};
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<EHALF>("TX"),FloatCastOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),FloatCastOp<EHALF,float,ehalf,float,ehalf4,float4>);

REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<   float>("TY").TypeConstraint<bfloat16>("TX"),FloatCastOp<float,bfloat16,float,bhalf,float4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<bfloat16>("TY").TypeConstraint<   float>("TX"),FloatCastOp<bfloat16,float,bhalf,float,bhalf4,float4>);

template <typename T, typename V> bool DropoutForward(CUstream stream, uint SMs, T* y, char* m, const T* x,uint size, float keep_prob, float scale);
template <typename T, typename V> bool DropoutBackward(CUstream stream, uint SMs,T* dx, const char* m, const T* dy,uint size, float scale);
template <typename T, typename V> bool DropoutMaskForward(CUstream stream, uint SMs, T* y, const char* m, const T* x, uint size, float scale);

REGISTER_OP("Dropout")
    .Input("x: T")
    .Input("keep_prob: float")
    .Input("scale: float")
    .Output("y: T")
    .Output("m: int8")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Simple elementwise dropout.
Random source is clock + tid/bid + tousworthe generator.
)doc");

template <typename T, typename V1, typename V4>
class DropoutOp : public OpKernel {
 public:
  explicit DropoutOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {}
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor& x = ctx->input(0);
    float keep_prob = ctx->input(1).scalar<float>()();
    float scale     = ctx->input(2).scalar<float>()();

    uint size = x.shape().num_elements();

    OP_REQUIRES(ctx, (size & 3) == 0, errors::Internal("Dropout tensor size must be multiple of 4"));
    OP_REQUIRES(ctx, keep_prob >= 0.0f && keep_prob <= 1.0f, errors::Internal("Dropout: 0 <= keep_prob <= 1"));

    Tensor* y = nullptr;
    Tensor* m = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x.shape(), &m));

          V1* y_ptr = (V1*)y->flat<T>().data();
        char* m_ptr = (char*)m->flat<int8>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    DropoutForward<V1,V4>(stream, SMs_, y_ptr, m_ptr, x_ptr, size, keep_prob, scale);
  }
  uint SMs_;
};
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<float>("T").HostMemory("keep_prob").HostMemory("scale"),DropoutOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("keep_prob").HostMemory("scale"),DropoutOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("keep_prob").HostMemory("scale"),DropoutOp<BHALF,bhalf,bhalf4>);

REGISTER_OP("DropoutMask")
    .Input("x: T")
    .Input("m: int8")
    .Input("scale: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Simple elementwise dropout with previously computed mask.
)doc");

template <typename T, typename V1, typename V4>
class DropoutMaskOp : public OpKernel {
 public:
  explicit DropoutMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {}
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor& x = ctx->input(0);
    const Tensor& m = ctx->input(1);
    float scale     = ctx->input(2).scalar<float>()();

    uint size = x.shape().num_elements();

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

            V1* y_ptr = (      V1*)y->flat<T>().data();
    const   V1* x_ptr = (const V1*)x.flat<T>().data();
    const char* m_ptr = (const char*)m.flat<int8>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    DropoutMaskForward<V1,V4>(stream, SMs_, y_ptr, m_ptr, x_ptr, size, scale);
  }
  uint SMs_;
};
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<float>("T").HostMemory("scale"),DropoutMaskOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("scale"),DropoutMaskOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("scale"),DropoutMaskOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("DropoutGrad")
    .Input("dy: T")
    .Input("m: int8")
    .Input("scale: float")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Simple elementwise dropout gradient.
)doc");

template <typename T, typename V1, typename V4>
class DropoutGradOp : public OpKernel {
 public:
  explicit DropoutGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {}
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor& dy = ctx->input(0);
    const Tensor& m  = ctx->input(1);
    float scale      = ctx->input(2).scalar<float>()();

    uint size = dy.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));

            V1* dx_ptr = (      V1*)dx->flat<T>().data();
    const   V1* dy_ptr = (const V1*)dy.flat<T>().data();
    const char* m_ptr  = (const char*)m.flat<int8>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    DropoutBackward<V1,V4>(stream, SMs_, dx_ptr, m_ptr, dy_ptr, size, scale);
  }
  uint SMs_;
};
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<float>("T").HostMemory("scale"),DropoutGradOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("scale"),DropoutGradOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("scale"),DropoutGradOp<BHALF,bhalf,bhalf4>);

template <typename T, typename V> bool AddN(CUstream stream, struct plist8<T>* x, T* z, int size, int params);

REGISTER_OP("AddN8")
    .Input("x8: N * T")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("N: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
  y = x[0] + x[1] + x[2] ... x[7]
)doc");

template <typename T, typename V1, typename V4>
class AddN8Op : public OpKernel {
 public:
  explicit AddN8Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x0 = ctx->input(0);

    int params = ctx->num_inputs();
    int size   = x0.shape().num_elements();

    struct plist8<V1> x8;
    for (int i = 0; i < params; ++i)
        x8.a[i] = (const V1*)ctx->input(i).flat<T>().data();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x0.shape(), &z));
    V1* z_ptr = (V1*)z->flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    AddN<V1,V4>(stream, &x8, z_ptr, size, params);
  }
};
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<float>("T"),AddN8Op<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),AddN8Op<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),AddN8Op<BHALF,bhalf,bhalf4>);


template <typename T, typename V> bool EW_Bias_Relu(CUstream stream, T* y, const T* x, const float* b, uint N, uint K, bool relu);
template <typename T, typename V> bool EW_Bias_Relu_Grad(CUstream stream, float* db, float* db_partial, T* dx, const T* dy, const T* y, uint gridN, uint gridK, uint vec, uint width, uint N, uint K, bool relu, bool partials);
void EW_Bias_Relu_Grad_Partial(bool partials, uint N, uint K, uint *gridN, uint *gridK, uint *vec, uint *width);

REGISTER_OP("BiasRelu")
    .Input("x: T")
    .Input("b: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("relu: bool = true")
    .Attr("atomics: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
bias op (with optional fused relu) for NHWC or NC layouts
very robust to odd shapes (like small C and large N)
)doc");

template <typename T, typename V1, typename V4>
class BiasReluOp : public OpKernel {
 public:
  explicit BiasReluOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("relu",  &relu_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench", &bench_ ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& b = ctx->input(1);

    int rank = x.dims();
    int K    = x.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    OP_REQUIRES(ctx, K == b.shape().num_elements(), errors::InvalidArgument("BiasRelu missmatched channels"));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

             V1* y_ptr = (V1*)y->flat<T>().data();
    const    V1* x_ptr = (const V1*)x.flat<T>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "BiasRelu     (%7d,%7d,%d)", N, K, (uint)sizeof(V1));
      bench = new Benchmark(stream, bench_string, 2*N*K*sizeof(V1) + K*sizeof(float), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EW_Bias_Relu<V1,V4>(stream, y_ptr, x_ptr, b_ptr, N, K, relu_);

    if (bench) delete bench;
  }
  bool relu_;
  int bench_;
};
REGISTER_KERNEL_BUILDER(Name("BiasRelu").Device(DEVICE_GPU).TypeConstraint<float>("T"),BiasReluOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("BiasRelu").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BiasReluOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("BiasRelu").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BiasReluOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("BiasReluGrad")
    .Input("dy: T")
    .Input("y: T")
    .Input("b: float")
    .Output("dx: T")
    .Output("db: float")
    .Output("temp: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("atomics: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
bias op grad with fused relu for NHWC or NC layouts
very robust to odd shapes (like small C and large N)
)doc");

template <typename T, typename V1, typename V4>
class BiasReluGradOp : public OpKernel {
 public:
  explicit BiasReluGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), N_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",   &bench_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("atomics", &atomics_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor&  y = ctx->input(1);
    const Tensor&  b = ctx->input(2);

    uint rank = dy.dims();
    uint K    = dy.dim_size(--rank);
    uint N    = 1;
    while (rank > 0) N *= dy.dim_size(--rank);

    Tensor* dx; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));
    Tensor* db; OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  b.shape(), &db));

    if (N_ != N)
    {
      EW_Bias_Relu_Grad_Partial(!atomics_, N, K, &gridN_, &gridK_, &vec_, &width_);
      N_ = N;
    }
    float* partial_ptr = nullptr;
    if (gridN_ > 1 && !atomics_)
    {
      Tensor* p; OP_REQUIRES_OK(ctx, ctx->allocate_output(2,  TensorShape({gridN_, K}), &p));
      partial_ptr = p->flat<float>().data();
    }
    else
    {
      Tensor* p; OP_REQUIRES_OK(ctx, ctx->allocate_output(2,  TensorShape(), &p));
    }

       float* db_ptr = (   float*)db->flat<float>().data();
          V1* dx_ptr = (      V1*)dx->flat<T>().data();
    const V1* dy_ptr = (const V1*)dy.flat<T>().data();
    const V1*  y_ptr = (const V1*)y.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "BiasReluGrad (%7d,%7d,%d) (gn:%3d gk:%3d v:%d w:%2d)", N, K, (uint)sizeof(V1), gridN_, gridK_, vec_, width_);
      bench = new Benchmark(stream, bench_string, 3*N*K*sizeof(V1) + K*sizeof(float), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EW_Bias_Relu_Grad<V1,V4>(stream, db_ptr, partial_ptr, dx_ptr, dy_ptr, y_ptr, gridN_, gridK_, vec_, width_, N, K, true, !atomics_);

    if (bench) delete bench;
  }
  int bench_;
  uint gridN_, gridK_, vec_, width_, N_;
  bool atomics_;
};
REGISTER_KERNEL_BUILDER(Name("BiasReluGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),BiasReluGradOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("BiasReluGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BiasReluGradOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("BiasReluGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BiasReluGradOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("BiasGrad")
    .Input("dy: T")
    .Input("b: float")
    .Output("db: float")
    .Output("temp: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("atomics: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
bias op grad (without fused relu) for NHWC or NC layouts
very robust to odd shapes (like small C and large N)
)doc");

template <typename T, typename V1, typename V4>
class BiasGradOp : public OpKernel {
 public:
  explicit BiasGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), N_(0) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",   &bench_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("atomics", &atomics_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor& b  = ctx->input(1);

    uint rank = dy.dims();
    uint K    = dy.dim_size(--rank);
    uint N    = 1;
    while (rank > 0) N *= dy.dim_size(--rank);

    Tensor* db; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b.shape(), &db));

    if (N_ != N)
    {
      EW_Bias_Relu_Grad_Partial(!atomics_, N, K, &gridN_, &gridK_, &vec_, &width_);
      N_ = N;
       // if (gridN_ > 1)
       //  OP_REQUIRES_OK(ctx, ctx->allocate_persistent(DT_FLOAT, TensorShape({gridN_, K}), &partial_, NULL));
    }
    // float* partial_ptr =  partial_.AccessTensor(ctx)->flat<float>().data(); // gridN_ > 1 ?: nullptr;

    float* partial_ptr = nullptr;
    if (gridN_ > 1 && !atomics_)
    {
      Tensor* p; OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  TensorShape({gridN_, K}), &p));
      partial_ptr = p->flat<float>().data();
    }
    else
    {
      Tensor* p; OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  TensorShape(), &p));
    }
    // //Tensor* Partial = nullptr;
    // //TensorShape shapeP;
    // Tensor Partial;
    // if (gridN_ > 1)
    // {
    //   // TensorShape shapeP;
    //   // shapeP.AddDim(gridN_);
    //   // shapeP.AddDim(K);
    //   OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({gridN_, K}), &Partial));
    //   partial_ptr = Partial.flat<float>().data();
    //}

       float* db_ptr = (   float*)db->flat<float>().data();
    const V1* dy_ptr = (const V1*)dy.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "BiasGrad     (%7d,%7d,%d) (gn:%3d gk:%3d v:%d w:%2d)", N, K, (uint)sizeof(V1), gridN_, gridK_, vec_, width_);
      bench = new Benchmark(stream, bench_string, N*K*sizeof(V1) + K*sizeof(float), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EW_Bias_Relu_Grad<V1,V4>(stream, db_ptr, partial_ptr, nullptr, dy_ptr, nullptr, gridN_, gridK_, vec_, width_, N, K, false, !atomics_);

    if (bench) delete bench;
  }
  int bench_;
  uint gridN_, gridK_, vec_, width_, N_;
  bool atomics_;
  //PersistentTensor partial_;
};
REGISTER_KERNEL_BUILDER(Name("BiasGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),BiasGradOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("BiasGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BiasGradOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("BiasGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BiasGradOp<BHALF,bhalf,bhalf4>);


template <typename T, typename TA> bool EW_Fancy_Gather(CUstream stream, T* y, const TA* a, const T* x, uint dim0, uint dim1, uint dim2);
template <typename T, typename TA> bool EW_Fancy_Gather_Grad(CUstream stream, T* dx, const TA* a, const T* dy, uint dim0, uint dim1, uint dim2);

REGISTER_OP("FancyGather")
    .Input("x: T")
    .Input("a: TA")
    .Output("y: T")
    .Attr("T: {int32, half, float, bfloat16}")
    .Attr("TA: { int32 }")
    .Attr("idx_dim: int = 1")
    .SetShapeFn([](InferenceContext* ctx) {

      ShapeHandle x = ctx->input(0);
      ShapeHandle a = ctx->input(1);
      int rankX = ctx->Rank(x);
      int rankA = ctx->Rank(a);

      std::vector<DimensionHandle> dims;
      for (int i = 0; i < rankX; ++i)
        if (i != rankA)
          dims.emplace_back(ctx->Dim(x, i));

      ctx->set_output(0, ctx->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Fancy Gather.
)doc");

template <typename T, typename V1, typename TA>
class FancyGatherOp : public OpKernel {
 public:
  explicit FancyGatherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& a = ctx->input(1);

    int rankX = x.dims();
    int rankA = a.dims();
    int dim1  = x.dim_size(rankA);
    int dim0  = 1, dim2 = 1;
    TensorShape shape;
    for (int i = 0; i < rankX; ++i)
    {
           if (i < rankA) dim0 *= x.dim_size(i);
      else if (i > rankA) dim2 *= x.dim_size(i);

      if (i != rankA)
        shape.AddDim(x.dim_size(i));
    }

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &y));

          V1* y_ptr = (      V1*)y->flat<T>().data();
    const TA* a_ptr = (const TA*)a.flat<TA>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Fancy_Gather<V1,TA>(stream, y_ptr, a_ptr, x_ptr, dim0, dim1, dim2);
  }
};
REGISTER_KERNEL_BUILDER(Name("FancyGather").Device(DEVICE_GPU).TypeConstraint<int32>("T").TypeConstraint<int>("TA"),FancyGatherOp<int32,  int,int>);
REGISTER_KERNEL_BUILDER(Name("FancyGather").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").TypeConstraint<int>("TA"),FancyGatherOp<FLOAT,float,int>);
REGISTER_KERNEL_BUILDER(Name("FancyGather").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<int>("TA"),FancyGatherOp<EHALF,ehalf,int>);
REGISTER_KERNEL_BUILDER(Name("FancyGather").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<int>("TA"),FancyGatherOp<BHALF,bhalf,int>);


REGISTER_OP("FancyGatherGrad")
    .Input("dy: T")
    .Input("a: TA")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("TA: { int32 }")
    .Attr("idx_dim: int = 1")
    .SetShapeFn([](InferenceContext* ctx) {

      int idx_dim;
      TF_RETURN_IF_ERROR(ctx->GetAttr("idx_dim", &idx_dim));

      ShapeHandle dy = ctx->input(0);
      ShapeHandle  a = ctx->input(1);
      int rankY = ctx->Rank(dy);
      int rankA = ctx->Rank(a);

      std::vector<DimensionHandle> dims;
      for (int i = 0; i < rankA; ++i)
        dims.emplace_back(ctx->Dim(dy, i));

      dims.emplace_back(ctx->MakeDim(idx_dim));

      for (int i = rankA; i < rankY; ++i)
        dims.emplace_back(ctx->Dim(dy, i));

      ctx->set_output(0, ctx->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
Fancy Gather.
)doc");

template <typename T, typename V1, typename TA>
class FancyGatherGradOp : public OpKernel {
 public:
  explicit FancyGatherGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("idx_dim", &idx_dim_ ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor&  a = ctx->input(1);

    int rankY = dy.dims();
    int rankA = a.dims();
    int dim1  = idx_dim_;
    int dim0  = 1, dim2 = 1;
    TensorShape shape;
    for (int i = 0; i < rankA; ++i)
    {
      dim0 *= dy.dim_size(i);
      shape.AddDim(dy.dim_size(i));
    }
    shape.AddDim(idx_dim_);
    for (int i = rankA; i < rankY; ++i)
    {
      dim2 *= dy.dim_size(i);
      shape.AddDim(dy.dim_size(i));
    }
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &dx));

          V1* dx_ptr = (      V1*)dx->flat<T>().data();
    const V1* dy_ptr = (const V1*)dy.flat<T>().data();
    const TA*  a_ptr = (const TA*)a.flat<TA>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Fancy_Gather_Grad<V1,TA>(stream, dx_ptr, a_ptr, dy_ptr, dim0, dim1, dim2);
  }
  int idx_dim_;
};
REGISTER_KERNEL_BUILDER(Name("FancyGatherGrad").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").TypeConstraint<int>("TA"),FancyGatherGradOp<FLOAT,float,int>);
REGISTER_KERNEL_BUILDER(Name("FancyGatherGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<int>("TA"),FancyGatherGradOp<EHALF,ehalf,int>);
REGISTER_KERNEL_BUILDER(Name("FancyGatherGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<int>("TA"),FancyGatherGradOp<BHALF,bhalf,int>);



template <typename T, typename TA> bool EW_Reduce_Max(CUstream stream, T* y, TA* a, const T* x, uint dim0, uint dim1, uint dim2);
template <typename T, typename TA> bool EW_Reduce_Max_Grad(CUstream stream, T* dx, const TA* a, const T* dy, uint dim0, uint dim1, uint dim2);

REGISTER_OP("ReduceMax")
    .Input("x: T")
    .Output("y: T")
    .Output("a: TA")
    .Attr("T: {half, float, bfloat16}")
    .Attr("TA: {uint8, uint16}")
    .Attr("axis: int")
    .Attr("keepdims: bool = false")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {

      int axis; bool keep_dims;
      TF_RETURN_IF_ERROR(ctx->GetAttr("axis",     &axis     ));
      TF_RETURN_IF_ERROR(ctx->GetAttr("keepdims", &keep_dims));

      ShapeHandle x = ctx->input(0);
      int rank = ctx->Rank(x);

      if (axis == rank-1)
        return errors::InvalidArgument("reductions on last axis not currently supported.");

      std::vector<DimensionHandle> dims;
      for (int i = 0; i < rank; ++i)
      {
        if (i == axis)
        {
          if (keep_dims)
            dims.emplace_back(ctx->MakeDim(1));
        }
        else
          dims.emplace_back(ctx->Dim(x, i));
      }
      ShapeHandle y = ctx->MakeShape(dims);
      ctx->set_output(0, y);
      ctx->set_output(1, y);
      return Status::OK();
    })
    .Doc(R"doc(
Reduce Max over a single column.
)doc");

template <typename T, typename V1, typename TA>
class ReduceMaxOp : public OpKernel {
 public:
  explicit ReduceMaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keepdims", &keep_dims_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_     ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int rank = x.dims();
    int dim1 = x.dim_size(axis_);
    int dim0 = 1, dim2 = 1;
    TensorShape shape;
    for (int i = 0; i < rank; ++i)
    {
           if (i < axis_) dim0 *= x.dim_size(i);
      else if (i > axis_) dim2 *= x.dim_size(i);

      if (i == axis_)
      {
        if (keep_dims_)
          shape.AddDim(1);
      }
      else
        shape.AddDim(x.dim_size(i));
    }

    Tensor* y = nullptr;
    Tensor* a = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shape, &a));

          V1* y_ptr = (      V1*)y->flat<T>().data();
          TA* a_ptr = (      TA*)a->flat<TA>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Reduce_Max<V1,TA>(stream, y_ptr, a_ptr, x_ptr, dim0, dim1, dim2);

  }
  bool keep_dims_;
  int bench_, axis_;
};
REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxOp<float,float,unsigned char>);
REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxOp<EHALF,ehalf,unsigned char>);
REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxOp<BHALF,bhalf,unsigned char>);

REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<ushort>("TA"),ReduceMaxOp<float,float,ushort>);
REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<ushort>("TA"),ReduceMaxOp<EHALF,ehalf,ushort>);
REGISTER_KERNEL_BUILDER(Name("ReduceMax").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<ushort>("TA"),ReduceMaxOp<BHALF,bhalf,ushort>);


REGISTER_OP("ReduceMaxGrad")
    .Input("dy: T")
    .Input("a: TA")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("TA: {uint8, uint16}")
    .Attr("axis: int")
    .Attr("axis_size: int")
    .Attr("keepdims: bool = false")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {

      int axis, axis_size; bool keep_dims;
      TF_RETURN_IF_ERROR(ctx->GetAttr("axis",      &axis     ));
      TF_RETURN_IF_ERROR(ctx->GetAttr("axis_size", &axis_size));
      TF_RETURN_IF_ERROR(ctx->GetAttr("keepdims",  &keep_dims));

      ShapeHandle dy = ctx->input(0);
      int rank = ctx->Rank(dy);

      std::vector<DimensionHandle> dims;
      for (int i = 0; i < rank; ++i)
      {
        if (i == axis)
        {
          dims.emplace_back(ctx->MakeDim(axis_size));
          if (!keep_dims)
            dims.emplace_back(ctx->Dim(dy, i));
        }
        else
          dims.emplace_back(ctx->Dim(dy, i));
      }
      ShapeHandle dx = ctx->MakeShape(dims);
      ctx->set_output(0, dx);
      return Status::OK();
    })
    .Doc(R"doc(
Reduce Max over a single column (gradient).
)doc");

template <typename T, typename V1, typename TA>
class ReduceMaxGradOp : public OpKernel {
 public:
  explicit ReduceMaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",      &axis_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis_size", &axis_size_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("keepdims",  &keep_dims_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",     &bench_     ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor&  a = ctx->input(1);

    int rank = dy.dims();
    int dim1 = axis_size_;
    int dim0 = 1, dim2 = 1;
    TensorShape shape;
    for (int i = 0; i < rank; ++i)
    {
      if (i == axis_)
      {
        shape.AddDim(axis_size_);
        if (!keep_dims_)
        {
          shape.AddDim(dy.dim_size(i));
          dim2 *= dy.dim_size(i);
        }
      }
      else
      {
             if (i < axis_) dim0 *= dy.dim_size(i);
        else if (i > axis_) dim2 *= dy.dim_size(i);
        shape.AddDim(dy.dim_size(i));
      }
    }

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &dx));

          V1* dx_ptr = (      V1*)dx->flat<T>().data();
    const V1* dy_ptr = (const V1*)dy.flat<T>().data();
    const TA*  a_ptr = (const TA*)a.flat<TA>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    EW_Reduce_Max_Grad<V1,TA>(stream, dx_ptr, a_ptr, dy_ptr, dim0, dim1, dim2);

  }
  bool keep_dims_;
  int bench_, axis_, axis_size_;
};
REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxGradOp<float,float,unsigned char>);
REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxGradOp<EHALF,ehalf,unsigned char>);
REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<unsigned char>("TA"),ReduceMaxGradOp<BHALF,bhalf,unsigned char>);

REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<float>("T").TypeConstraint<ushort>("TA"),ReduceMaxGradOp<float,float,ushort>);
REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").TypeConstraint<ushort>("TA"),ReduceMaxGradOp<EHALF,ehalf,ushort>);
REGISTER_KERNEL_BUILDER(Name("ReduceMaxGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").TypeConstraint<ushort>("TA"),ReduceMaxGradOp<BHALF,bhalf,ushort>);


