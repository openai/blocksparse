
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


template <typename V> bool Gemm_TN(CUstream stream, uint SMs, int major, float* u, const V* x, const V* e, uint C, uint K, uint N);

REGISTER_OP("DwMatmulLargeN")
    .Input("x: T")
    .Input("e: T")
    .Output("u: float")
    .Attr("T: {float, half}")
    .SetShapeFn([](InferenceContext* ctx) {

      ShapeHandle x = ctx->input(0);
      ShapeHandle e = ctx->input(1);

      if (ctx->RankKnown(x) && ctx->RankKnown(e))
        ctx->set_output(0, ctx->MakeShape( {
          ctx->Dim(x, ctx->Rank(x)-1), // C
          ctx->Dim(e, ctx->Rank(e)-1)  // K
        } ));
      else
        ctx->set_output(0, ctx->UnknownShape());

      return Status::OK();
    })
    .Doc(R"doc(
Row Major Matmul: C = A.T x B
Special kernel for very large grad weight reductions (very large effective minibatch).
Mainly for boosting accuracy by also for better spanning over SMs
)doc");

template <typename T, typename V4>
class DwMatmulLargeNOp : public OpKernel {
 public:
  explicit DwMatmulLargeNOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), major_(0) {}

  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_  = GetCountSMsVersion(&major_, NULL);

    const Tensor& x = ctx->input(0);
    const Tensor& e = ctx->input(1);

    OP_REQUIRES(ctx, e.dims() == x.dims(), errors::InvalidArgument("Mismatched Shapes"));

    int last_dim = x.dims() - 1;
    uint C = x.dim_size(last_dim);
    uint K = e.dim_size(last_dim);
    uint N = 1;
    for (int i = 0; i < last_dim; i++)
    {
      OP_REQUIRES(ctx, x.dim_size(i) == e.dim_size(i), errors::InvalidArgument("Mismatched Shapes"));
      N *= x.dim_size(i);
    }
    OP_REQUIRES(ctx, (C & 3) == 0 && (K & 3) == 0, errors::InvalidArgument("Channel dims must be multiple of 4"));
    OP_REQUIRES(ctx, (N & 31) == 0, errors::InvalidArgument("Minibatch dim must be multiple of 32"));

    Tensor* u; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({C, K}), &u));

       float* u_ptr = u->flat<float>().data();
    const V4* x_ptr = (const V4*)x.flat<T>().data();
    const V4* e_ptr = (const V4*)e.flat<T>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Gemm_TN(stream, SMs_, major_, u_ptr, x_ptr, e_ptr, C, K, N);
  }
  int SMs_, major_;
};
REGISTER_KERNEL_BUILDER(Name("DwMatmulLargeN").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), DwMatmulLargeNOp<FLOAT,float4>);
REGISTER_KERNEL_BUILDER(Name("DwMatmulLargeN").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), DwMatmulLargeNOp<EHALF,ehalf4>);
