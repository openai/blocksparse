
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

template <typename TY, typename TX> bool L2NormalizeKCTRS(CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K);
template <typename TY, typename TX> bool L2NormalizeCKTRS(CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template <typename TY, typename TX> bool L2NormalizeCK   (CUstream stream, TY* y, float* sum_sqr_x, const TX* x, const float* g, const int* lut, float epsilon, int K, int shared, int bsize_);

Status L2NormalizeShape(InferenceContext* ctx)
{
  int K; TF_RETURN_IF_ERROR(ctx->GetAttr("K", &K));

  ctx->set_output(0, ctx->input(0));
  ctx->set_output(1, ctx->Vector(K));

  return Status::OK();
}

REGISTER_OP("L2NormalizeKCTRS")
    .Input("x: TX")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse convolution with KCTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeCKTRS")
    .Input("x: TX")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("TRS: int")
    .Attr("magic_TRS: int")
    .Attr("shift_TRS: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse convolution with CKTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeCK")
    .Input("x: TX")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("shared: int")
    .Attr("bsize: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse matmul with CK weight layout.
)doc");


template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeKCTRSOp : public OpKernel {
 public:
  explicit L2NormalizeKCTRSOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "K",       &K_));
  }

  virtual void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& lut = ctx->input(1);

    TensorShape sum_shape({ K_ });

    Tensor* y = nullptr;
    Tensor* sum_sqr_x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, sum_shape, &sum_sqr_x));

             VY*     y_ptr = (VY*)y->flat<TY>().data();
          float* sum_x_ptr = sum_sqr_x->flat<float>().data();
    const    VX*     x_ptr = (const VX*)x.flat<TX>().data();
    const   int*   lut_ptr = lut.flat<int32>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    this->L2Normalize(stream, y_ptr, sum_x_ptr, x_ptr, lut_ptr, epsilon_, K_);
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const int* lut, float epsilon, int K) {

    return L2NormalizeKCTRS<VY,VX>(stream, y, sum_sqr_x, x, 0, lut, epsilon, K);
  }
  float epsilon_;
  int   K_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeCKTRSOp : public L2NormalizeKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeCKTRSOp(OpKernelConstruction* ctx) : L2NormalizeKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "TRS",       &TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_TRS", &magic_TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_TRS", &shift_TRS_ ));
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const int* lut, float epsilon, int K) {
    return L2NormalizeCKTRS<VY,VX>(stream, y, sum_sqr_x, x, 0, lut, epsilon, K, TRS_, magic_TRS_, shift_TRS_);
  }
  int TRS_, magic_TRS_, shift_TRS_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeCKOp : public L2NormalizeKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeCKOp(OpKernelConstruction* ctx) : L2NormalizeKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared", &shared_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const int* lut, float epsilon, int K) {
    return L2NormalizeCK<VY,VX>(stream, y, sum_sqr_x, x, 0, lut, epsilon, K, shared_, bsize_);
  }
  int shared_, bsize_;
};


REGISTER_KERNEL_BUILDER(Name("L2NormalizeKCTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeKCTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeKCTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeKCTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeKCTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeKCTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeCKTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeCKTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeCKTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeCKTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeCKTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeCKTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeCK").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeCKOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeCKOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeCKOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeCKOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeCKOp<BHALF,float,bhalf,float>);



REGISTER_OP("L2NormalizeGainKCTRS")
    .Input("x: TX")
    .Input("g: float")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse convolution with KCTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGainCKTRS")
    .Input("x: TX")
    .Input("g: float")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("TRS: int")
    .Attr("magic_TRS: int")
    .Attr("shift_TRS: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse convolution with CKTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGainCK")
    .Input("x: TX")
    .Input("g: float")
    .Input("lut: int32")
    .Output("y: TY")
    .Output("sum_sqr_x: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("shared: int")
    .Attr("bsize: int")
   .SetShapeFn(L2NormalizeShape)
    .Doc(R"doc(
l2_normalize for blocksparse matmul with CK weight layout.
)doc");

template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainKCTRSOp : public OpKernel {
 public:
  explicit L2NormalizeGainKCTRSOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "K",       &K_));
  }

  virtual void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& g   = ctx->input(1);
    const Tensor& lut = ctx->input(2);

    TensorShape sum_shape({ K_ });

    Tensor* y = nullptr;
    Tensor* sum_sqr_x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, sum_shape, &sum_sqr_x));

             VY*     y_ptr = (VY*)y->flat<TY>().data();
          float* sum_x_ptr = sum_sqr_x->flat<float>().data();
    const    VX*     x_ptr = (const VX*)x.flat<TX>().data();
    const float*     g_ptr = g.flat<float>().data();
    const   int*   lut_ptr = lut.flat<int32>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    this->L2Normalize(stream, y_ptr, sum_x_ptr, x_ptr, g_ptr, lut_ptr, epsilon_, K_);
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const float* g, const int* lut, float epsilon, int K) {

    return L2NormalizeKCTRS<VY,VX>(stream, y, sum_sqr_x, x, g, lut, epsilon, K);
  }
  float epsilon_;
  int   K_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainCKTRSOp : public L2NormalizeGainKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGainCKTRSOp(OpKernelConstruction* ctx) : L2NormalizeGainKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "TRS",       &TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_TRS", &magic_TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_TRS", &shift_TRS_ ));
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const float* g, const int* lut, float epsilon, int K) {
    return L2NormalizeCKTRS<VY,VX>(stream, y, sum_sqr_x, x, g, lut, epsilon, K, TRS_, magic_TRS_, shift_TRS_);
  }
  int TRS_, magic_TRS_, shift_TRS_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainCKOp : public L2NormalizeGainKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGainCKOp(OpKernelConstruction* ctx) : L2NormalizeGainKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared", &shared_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
  }
  virtual bool L2Normalize(CUstream stream, VY* y, float* sum_sqr_x, const VX* x, const float* g, const int* lut, float epsilon, int K) {
    return L2NormalizeCK<VY,VX>(stream, y, sum_sqr_x, x, g, lut, epsilon, K, shared_, bsize_);
  }
  int shared_, bsize_;
};

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainKCTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainKCTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainKCTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainKCTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainKCTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainKCTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCKTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainCKTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainCKTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCK").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainCKOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainCKOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainCKOp<BHALF,float,bhalf,float>);



/////////////////////////////////////// Gradients ///////////////////////////////////////////


template <typename TY, typename TX> bool L2NormalizeGradKCTRS(CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K);
template <typename TY, typename TX> bool L2NormalizeGradCKTRS(CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int TRS, int magic_TRS, int shift_TRS);
template <typename TY, typename TX> bool L2NormalizeGradCK   (CUstream stream, TX* grad_x, float* grad_g, const TY* grad_y, const TX* x, const float* g, const float* sum_sqr_x_p, const int* lut, float epsilon, int K, int shared, int bsize);


Status L2NormalizeGradShape(InferenceContext* ctx)
{
  ctx->set_output(0, ctx->input(1));
  return Status::OK();
}

REGISTER_OP("L2NormalizeGradKCTRS")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
   .SetShapeFn(L2NormalizeGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse convolution with KCTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGradCKTRS")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("TRS: int")
    .Attr("magic_TRS: int")
    .Attr("shift_TRS: int")
   .SetShapeFn(L2NormalizeGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse convolution with CKTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGradCK")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("shared: int")
    .Attr("bsize: int")
   .SetShapeFn(L2NormalizeGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse matmul with CK weight layout.
)doc");

template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGradKCTRSOp : public OpKernel {
 public:
  explicit L2NormalizeGradKCTRSOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "K",       &K_));
  }
  virtual void Compute(OpKernelContext* ctx) override {

    const Tensor& grad_y = ctx->input(0);
    const Tensor&      x = ctx->input(1);
    const Tensor&  sum_x = ctx->input(2);
    const Tensor&    lut = ctx->input(3);

    Tensor* grad_x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &grad_x));

    const    VY* grad_y_ptr = (const VY*)grad_y.flat<TY>().data();
    const    VX*      x_ptr = (const VX*)x.flat<TX>().data();
    const float*  sum_x_ptr = sum_x.flat<float>().data();
    const   int*    lut_ptr = lut.flat<int32>().data();
             VX* grad_x_ptr = (VX*)grad_x->flat<TX>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    this->L2NormalizeGrad(stream, grad_x_ptr, grad_y_ptr, x_ptr, sum_x_ptr, lut_ptr, epsilon_, K_);
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, const VY* grad_y, const VX* x, const float* sum_sqr_x, const int* lut, float epsilon, int K) {

    return L2NormalizeGradKCTRS<VY,VX>(stream, grad_x, 0, grad_y, x, 0, sum_sqr_x, lut, epsilon, K);
  }
  float epsilon_;
  int   K_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGradCKTRSOp : public L2NormalizeGradKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGradCKTRSOp(OpKernelConstruction* ctx) : L2NormalizeGradKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "TRS",       &TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_TRS", &magic_TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_TRS", &shift_TRS_ ));
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, const VY* grad_y, const VX* x, const float* sum_sqr_x, const int* lut, float epsilon, int K) {
    return L2NormalizeGradCKTRS<VY,VX>(stream, grad_x, 0, grad_y, x, 0, sum_sqr_x, lut, epsilon, K, TRS_, magic_TRS_, shift_TRS_);
  }
  int TRS_, magic_TRS_, shift_TRS_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGradCKOp : public L2NormalizeGradKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGradCKOp(OpKernelConstruction* ctx) : L2NormalizeGradKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared", &shared_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, const VY* grad_y, const VX* x, const float* sum_sqr_x, const int* lut, float epsilon, int K) {
    return L2NormalizeGradCK<VY,VX>(stream, grad_x, 0, grad_y, x, 0, sum_sqr_x, lut, epsilon, K, shared_, bsize_);
  }
  int shared_, bsize_;
};

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradKCTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGradKCTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGradKCTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradKCTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGradKCTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradKCTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCKTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGradCKTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGradCKTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCK").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGradCKOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGradCKOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGradCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGradCKOp<BHALF,float,bhalf,float>);



Status L2NormalizeGainGradShape(InferenceContext* ctx)
{
  ctx->set_output(0, ctx->input(1));
  ctx->set_output(1, ctx->input(2));
  return Status::OK();
}

REGISTER_OP("L2NormalizeGainGradKCTRS")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("g: float")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Output("grad_g: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
   .SetShapeFn(L2NormalizeGainGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse convolution with KCTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGainGradCKTRS")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("g: float")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Output("grad_g: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("TRS: int")
    .Attr("magic_TRS: int")
    .Attr("shift_TRS: int")
   .SetShapeFn(L2NormalizeGainGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse convolution with CKTRS filter layout.
)doc");

REGISTER_OP("L2NormalizeGainGradCK")
    .Input("grad_y: TY")
    .Input("x: TX")
    .Input("g: float")
    .Input("sum_sqr_x: float")
    .Input("lut: int32")
    .Output("grad_x: TX")
    .Output("grad_g: float")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("epsilon: float")
    .Attr("K: int")
    .Attr("shared: int")
    .Attr("bsize: int")
   .SetShapeFn(L2NormalizeGradShape)
    .Doc(R"doc(
l2_normalize gradient for blocksparse matmul with CK weight layout.
)doc");

template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainGradKCTRSOp : public OpKernel {
 public:
  explicit L2NormalizeGainGradKCTRSOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "K",       &K_));
  }
  virtual void Compute(OpKernelContext* ctx) override {

    const Tensor& grad_y = ctx->input(0);
    const Tensor&      x = ctx->input(1);
    const Tensor&      g = ctx->input(2);
    const Tensor&  sum_x = ctx->input(3);
    const Tensor&    lut = ctx->input(4);

    Tensor* grad_x = nullptr;
    Tensor* grad_g = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &grad_x));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &grad_g));

    const    VY* grad_y_ptr = (const VY*)grad_y.flat<TY>().data();
    const    VX*      x_ptr = (const VX*)x.flat<TX>().data();
    const float*      g_ptr = g.flat<float>().data();
    const float*  sum_x_ptr = sum_x.flat<float>().data();
    const   int*    lut_ptr = lut.flat<int32>().data();
             VX* grad_x_ptr = (VX*)grad_x->flat<TX>().data();
          float* grad_g_ptr = grad_g->flat<float>().data();

    CUstream stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    this->L2NormalizeGrad(stream, grad_x_ptr, grad_g_ptr, grad_y_ptr, x_ptr, g_ptr, sum_x_ptr, lut_ptr, epsilon_, K_);
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, float* grad_g, const VY* grad_y, const VX* x, const float* g, const float* sum_sqr_x, const int* lut, float epsilon, int K) {

    return L2NormalizeGradKCTRS<VY,VX>(stream, grad_x, grad_g, grad_y, x, g, sum_sqr_x, lut, epsilon, K);
  }
  float epsilon_;
  int   K_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainGradCKTRSOp : public L2NormalizeGainGradKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGainGradCKTRSOp(OpKernelConstruction* ctx) : L2NormalizeGainGradKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(      "TRS",       &TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_TRS", &magic_TRS_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_TRS", &shift_TRS_ ));
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, float* grad_g, const VY* grad_y, const VX* x, const float* g, const float* sum_sqr_x, const int* lut, float epsilon, int K) {
    return L2NormalizeGradCKTRS<VY,VX>(stream, grad_x, grad_g, grad_y, x, g, sum_sqr_x, lut, epsilon, K, TRS_, magic_TRS_, shift_TRS_);
  }
  int TRS_, magic_TRS_, shift_TRS_;
};
template <typename TY, typename TX, typename VY, typename VX>
class L2NormalizeGainGradCKOp : public L2NormalizeGainGradKCTRSOp<TY,TX,VY,VX> {
 public:
  explicit L2NormalizeGainGradCKOp(OpKernelConstruction* ctx) : L2NormalizeGainGradKCTRSOp<TY,TX,VY,VX>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared", &shared_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
  }
  virtual bool L2NormalizeGrad(CUstream stream, VX* grad_x, float* grad_g, const VY* grad_y, const VX* x, const float* g, const float* sum_sqr_x, const int* lut, float epsilon, int K) {
    return L2NormalizeGradCK<VY,VX>(stream, grad_x, grad_g, grad_y, x, g, sum_sqr_x, lut, epsilon, K, shared_, bsize_);
  }
  int shared_, bsize_;
};


REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradKCTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradKCTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainGradKCTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradKCTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradKCTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainGradKCTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradKCTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradKCTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCKTRS").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKTRSOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainGradCKTRSOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCKTRS").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKTRSOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainGradCKTRSOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCKTRS").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKTRSOp<BHALF,float,bhalf,float>);

REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCK").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKOp<float,float,float,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<EHALF>("TX"),L2NormalizeGainGradCKOp<EHALF,EHALF,ehalf,ehalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCK").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKOp<EHALF,float,ehalf,float>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<BHALF>("TX"),L2NormalizeGainGradCKOp<BHALF,BHALF,bhalf,bhalf>);
REGISTER_KERNEL_BUILDER(Name("L2NormalizeGainGradCK").Device(DEVICE_GPU).TypeConstraint<BHALF>("TY").TypeConstraint<float>("TX"),L2NormalizeGainGradCKOp<BHALF,float,bhalf,float>);
