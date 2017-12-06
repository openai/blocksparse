
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

template <typename T> bool EdgeBiasForward(CUstream, const T*, const float*, const int*, int, int, int, int, int);
template <typename T> bool EdgeBiasBackward(CUstream, float*, const T*, const int*, int, int, int, int, int);

Status EdgeBiasShape(InferenceContext* ctx)
{
  int K, edges;
  std::vector<int32> MPQ;

  TF_RETURN_IF_ERROR(ctx->GetAttr("edges", &edges));
  TF_RETURN_IF_ERROR(ctx->GetAttr("K",     &K));
  TF_RETURN_IF_ERROR(ctx->GetAttr("MPQ",   &MPQ));

  ShapeHandle I, B;
  DimensionHandle dh;
  TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 3, &I));
  TF_RETURN_IF_ERROR(ctx->WithRank(       ctx->input(1), 2, &B));

  TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(B, 0),     K, &dh));
  TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(B, 1), edges, &dh));

  TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 1),     K, &dh));

  int rankI = ctx->Rank(I);

  if (rankI == 5)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[0], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[1], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 4), MPQ[2], &dh));
  }
  else if (rankI == 4)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[1], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[2], &dh));
  }
  else if (rankI == 3)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[2], &dh));
  }
  else
    return errors::InvalidArgument("EdgeBias requires an input rank between 3 and 5: ", rankI);

  ctx->set_output(0, ctx->input(0));
  return Status::OK();
}

REGISTER_OP("EdgeBias")
    .Input("x: T")
    .Input("b: float")
    .Input("bias_lut: int32")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("edges: int >= 0")
    .Attr("K: int >= 0")
    .Attr("MPQ: list(int) >= 3")
   .SetShapeFn(EdgeBiasShape)
    .Doc(R"doc(
Edge bias for Convolution.
)doc");


template <typename T, typename V>
class EdgeBiasOp : public OpKernel {
 public:
  explicit EdgeBiasOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("edges", &edges_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",     &K_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("MPQ",   &MPQ_  ));

    mpq_  = MPQ_[0] * MPQ_[1] * MPQ_[2];
    kmpq_ = K_ * mpq_;
  }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& b   = ctx->input(1);
    const Tensor& lut = ctx->input(2);

    int N = x.dim_size(0);

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    ctx->set_output(0, x);

    const V*     x_ptr = (const V*)x.flat<T>().data();
    const float* b_ptr = b.flat<float>().data();
    const int* lut_ptr = lut.flat<int32>().data();

    EdgeBiasForward<V>(stream, x_ptr, b_ptr, lut_ptr, edges_, mpq_, kmpq_, K_, N);
    // TODO: error check
  }

 private:

  int edges_, K_, mpq_, kmpq_;
  std::vector<int32> MPQ_;

};

REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<float>("T"),EdgeBiasOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EdgeBiasOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EdgeBiasOp<BHALF,bhalf>);


/////////////////////////////////////// Gradient ///////////////////////////////////////////


Status EdgeBiasGradShape(InferenceContext* ctx)
{
  int K, edges;
  std::vector<int32> MPQ;

  TF_RETURN_IF_ERROR(ctx->GetAttr("edges", &edges));
  TF_RETURN_IF_ERROR(ctx->GetAttr("K",     &K));
  TF_RETURN_IF_ERROR(ctx->GetAttr("MPQ",   &MPQ));

  ShapeHandle I;
  DimensionHandle dh;
  TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 3, &I));
  TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 1), K, &dh));

  int rankI = ctx->Rank(I);

  if (rankI == 5)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[0], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[1], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 4), MPQ[2], &dh));
  }
  else if (rankI == 4)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[1], &dh));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[2], &dh));
  }
  else if (rankI == 3)
  {
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[2], &dh));
  }
  else
    return errors::InvalidArgument("EdgeBiasGrad requires an input rank between 3 and 5: ", rankI);

  ctx->set_output(0, ctx->MakeShape({ ctx->Dim(I, 1), ctx->MakeDim(edges) }));// K x edeges

  return Status::OK();
}

REGISTER_OP("EdgeBiasGrad")
    .Input("grad_y: T")
    .Input("bias_lut: int32")
    .Output("grad_b: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("edges: int >= 0")
    .Attr("K: int >= 0")
    .Attr("MPQ: list(int) >= 3")
   .SetShapeFn(EdgeBiasGradShape)
    .Doc(R"doc(
Edge bias for Convolution.
)doc");


template <typename T, typename V>
class EdgeBiasGradOp : public OpKernel {
 public:
  explicit EdgeBiasGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("edges", &edges_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",     &K_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("MPQ",   &MPQ_  ));

    mpq_  = MPQ_[0] * MPQ_[1] * MPQ_[2];
    kmpq_ = K_ * mpq_;
  }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& grad_y = ctx->input(0);
    const Tensor& lut    = ctx->input(1);

    int N = grad_y.dim_size(0);

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    Tensor* grad_b = nullptr;
    TensorShape bias_shape({ K_, edges_ });

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, bias_shape, &grad_b));

          float* grad_b_ptr = grad_b->flat<float>().data();
    const     V* grad_y_ptr = (const V*)grad_y.flat<T>().data();
    const   int* lut_ptr    = lut.flat<int32>().data();

    EdgeBiasBackward<V>(stream, grad_b_ptr, grad_y_ptr, lut_ptr, edges_, mpq_, kmpq_, K_, N);
    // TODO: error check
  }

 private:

  int edges_, K_, mpq_, kmpq_;
  std::vector<int32> MPQ_;

};

REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),EdgeBiasGradOp<float,float>);
REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EdgeBiasGradOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EdgeBiasGradOp<BHALF,bhalf>);




