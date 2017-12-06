
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

template <typename T> bool BatchNormNCDHW_Inference(
  CUstream stream, T* y, const float* m, const float* v, const T* x, const float* g, const float* b,
  int N, int C, int DHW, float epsilon);

REGISTER_OP("BatchNormInferenceNCDHW")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Input("m: float")
    .Input("v: float")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("DHW: int")
    .Attr("eps: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
BatchNorm Inference NCDHW
)doc");

template <typename T, typename V>
class BatchNormInferenceNCDHWOp : public OpKernel {
 public:
  explicit BatchNormInferenceNCDHWOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eps", &eps_ ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& g = ctx->input(1);
    const Tensor& b = ctx->input(2);
    const Tensor& m = ctx->input(3);
    const Tensor& v = ctx->input(4);

    int N = x.dim_size(0);
    int C = x.dim_size(1);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

              V* y_ptr = (V*)y->flat<T>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* g_ptr = g.flat<float>().data();
    const float* b_ptr = b.flat<float>().data();
    const float* m_ptr = m.flat<float>().data();
    const float* v_ptr = v.flat<float>().data();

    CUstream stream = NULL; //AsCUDAStreamValue(ctx->op_device_context()->stream());

    BatchNormNCDHW_Inference<V>(stream, y_ptr, m_ptr, v_ptr, x_ptr, g_ptr, b_ptr, N, C, DHW_, eps_);
  }
  int   DHW_;
  float eps_;
};
REGISTER_KERNEL_BUILDER(Name("BatchNormInferenceNCDHW").Device(DEVICE_GPU).TypeConstraint<float>("T"),BatchNormInferenceNCDHWOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("BatchNormInferenceNCDHW").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BatchNormInferenceNCDHWOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("BatchNormInferenceNCDHW").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BatchNormInferenceNCDHWOp<BHALF,bhalf>);



template <typename T> bool BatchNormNCDHW_Forward(
  CUstream stream, T* y, float* m, float* v, const T* x, const float* g, const float* b,
  int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

REGISTER_OP("BatchNormNCDHW")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Output("y: T")
    .Output("m: float")
    .Output("v: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("DHW: int")
    .Attr("magic_DHW: int")
    .Attr("shift_DHW: int")
    .Attr("eps: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(1));
      ctx->set_output(2, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
BatchNorm NCDHW
)doc");

template <typename T, typename V>
class BatchNormNCDHWOp : public OpKernel {
 public:
  explicit BatchNormNCDHWOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eps", &eps_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_DHW", &magic_DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_DHW", &shift_DHW_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& g = ctx->input(1);
    const Tensor& b = ctx->input(2);

    int N = x.dim_size(0);
    int C = x.dim_size(1);

    Tensor* y = nullptr;
    Tensor* m = nullptr;
    Tensor* v = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &m));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, g.shape(), &v));

              V* y_ptr = (V*)y->flat<T>().data();
          float* m_ptr = m->flat<float>().data();
          float* v_ptr = v->flat<float>().data();
    const     V* x_ptr = (const V*)x.flat<T>().data();
    const float* g_ptr = g.flat<float>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = NULL; //AsCUDAStreamValue(ctx->op_device_context()->stream());

    BatchNormNCDHW_Forward<V>(stream, y_ptr, m_ptr, v_ptr, x_ptr, g_ptr, b_ptr, N, C, DHW_, magic_DHW_, shift_DHW_, eps_);
  }
  int DHW_, magic_DHW_, shift_DHW_;
  float eps_;
};
REGISTER_KERNEL_BUILDER(Name("BatchNormNCDHW").Device(DEVICE_GPU).TypeConstraint<float>("T"),BatchNormNCDHWOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("BatchNormNCDHW").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),BatchNormNCDHWOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("BatchNormNCDHW").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),BatchNormNCDHWOp<BHALF,bhalf>);


template <typename TX, typename TY> bool BatchNormNCDHW_Backward(
  CUstream stream, TY* dx, float* dg, float* db, const TY* dy, const TX* x, const float* g, const float* m, const float* v,
  int N, int C, int DHW, int magic_DHW, int shift_DHW, float epsilon);

REGISTER_OP("BatchNormGradNCDHW")
    .Input("dy: TY")
    .Input("x: TX")
    .Input("g: float")
    .Input("m: float")
    .Input("v: float")
    .Output("dx: TY")
    .Output("dg: float")
    .Output("db: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("DHW: int")
    .Attr("magic_DHW: int")
    .Attr("shift_DHW: int")
    .Attr("eps: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      ctx->set_output(2, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
BatchNorm Grad NCDHW
)doc");

template <typename TX, typename TY, typename VX, typename VY>
class BatchNormGradNCDHWOp : public OpKernel {
 public:
  explicit BatchNormGradNCDHWOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW", &DHW_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eps", &eps_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_DHW", &magic_DHW_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_DHW", &shift_DHW_));
  }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor&  x = ctx->input(1);
    const Tensor&  g = ctx->input(2);
    const Tensor&  m = ctx->input(3);
    const Tensor&  v = ctx->input(4);

    int N = x.dim_size(0);
    int C = x.dim_size(1);

    Tensor* dx = nullptr;
    Tensor* dg = nullptr;
    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &dg));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, g.shape(), &db));

             VY* dx_ptr = (VY*)dx->flat<TY>().data();
          float* dg_ptr = dg->flat<float>().data();
          float* db_ptr = db->flat<float>().data();
    const    VY* dy_ptr = (const VY*)dy.flat<TY>().data();
    const    VX*  x_ptr = (const VX*)x.flat<TX>().data();
    const float*  g_ptr = g.flat<float>().data();
    const float*  m_ptr = m.flat<float>().data();
    const float*  v_ptr = v.flat<float>().data();

    CUstream stream = NULL; //AsCUDAStreamValue(ctx->op_device_context()->stream());

    BatchNormNCDHW_Backward<VX,VY>(stream, dx_ptr, dg_ptr, db_ptr, dy_ptr, x_ptr, g_ptr, m_ptr, v_ptr, N, C, DHW_, magic_DHW_, shift_DHW_, eps_);
  }
  int   DHW_, magic_DHW_, shift_DHW_;
  float eps_;
};
REGISTER_KERNEL_BUILDER( Name("BatchNormGradNCDHW").Device(DEVICE_GPU).TypeConstraint<float>("TX").TypeConstraint<float>("TY"), BatchNormGradNCDHWOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER( Name("BatchNormGradNCDHW").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<EHALF>("TY"), BatchNormGradNCDHWOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER( Name("BatchNormGradNCDHW").Device(DEVICE_GPU).TypeConstraint<EHALF>("TX").TypeConstraint<float>("TY"), BatchNormGradNCDHWOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER( Name("BatchNormGradNCDHW").Device(DEVICE_GPU).TypeConstraint<BHALF>("TX").TypeConstraint<BHALF>("TY"), BatchNormGradNCDHWOp<BHALF,BHALF,bhalf,bhalf>);


