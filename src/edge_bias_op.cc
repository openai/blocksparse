
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


template <typename T> bool EdgeBiasForward (CUstream stream, T* y, const T* x, const float* g, const float* b, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout, bool inference);
template <typename T> bool EdgeBiasBackward(CUstream stream, T* dy, float* dg, float* db, const T* x, const float* g, const int* lut, uint edges, uint MPQ, uint K, uint N, int layout);

Status UnchangedShape(shape_inference::InferenceContext* ctx);


REGISTER_OP("EdgeBias")
    .Input("x: T")
    .Input("g: float")
    .Input("b: float")
    .Input("lut: int32")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("layout: int = 0")
    .Attr("entries: int = 0")
    .Attr("inference: bool = false")
    .Attr("bench: int = 0")
   .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Edge bias for Convolution.
layout: 0=NCHW, 1=NHWC
)doc");


template <typename T, typename V>
class EdgeBiasOp : public OpKernel {
 public:
  explicit EdgeBiasOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("layout",    &layout_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("entries",   &entries_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",     &bench_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("inference", &inference_));
  }

  void Compute(OpKernelContext* ctx) override {

    const Tensor& x   = ctx->input(0);
    const Tensor& g   = ctx->input(1);
    const Tensor& b   = ctx->input(2);
    const Tensor& lut = ctx->input(3);

    uint rank = x.dims();
    uint N    = x.dim_size(0);
    uint MPQ  = 1, K, edges;
    // NCHW
    if (layout_ == 0)
    {
      K = x.dim_size(1);
      for (int i = 2; i < rank; i++)
        MPQ *= x.dim_size(i);

      edges = b.dim_size(1);
    }
    // NHWC
    else
    {
      K = x.dim_size(rank-1);
      for (int i = 1; i < rank-1; i++)
        MPQ *= x.dim_size(i);

      edges = b.dim_size(0);
    }
    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    const V*     x_ptr = (V*)x.flat<T>().data();
    const float* g_ptr = g.flat<float>().data();
    const float* b_ptr = b.flat<float>().data();
    const int* lut_ptr = lut.flat<int32>().data();

    V* y_ptr;
    if (inference_)
    {
      // in place
      ctx->set_output(0, x);
      y_ptr = (V*)x_ptr;
    }
    else
    {
      Tensor* y;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
      y_ptr = (V*)y->flat<T>().data();
    }

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "EdgeBias     N:%3d,K:%3d,E:%2d L:%d", N, K, edges, layout_);
      bench = new Benchmark(stream, bench_string, 2*N*K*MPQ*sizeof(V) + 2*N*K*entries_*sizeof(V) + 2*K*edges*sizeof(float), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EdgeBiasForward<V>(stream, y_ptr, x_ptr, g_ptr, b_ptr, lut_ptr, edges, MPQ, K, N, layout_, inference_);

    if (bench) delete bench;
  }

 private:
  int layout_, bench_, entries_;
  bool inference_;
};

REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),EdgeBiasOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EdgeBiasOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EdgeBias").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EdgeBiasOp<BHALF,bhalf>);


/////////////////////////////////////// Gradient ///////////////////////////////////////////

REGISTER_OP("EdgeBiasGrad")
    .Input("dy: T")
    .Input("x: T")
    .Input("g: float")
    .Input("lut: int32")
    .Output("dx: T")
    .Output("dg: float")
    .Output("db: float")
    .Attr("T: {half, float, bfloat16}")
    .Attr("layout: int = 0")
    .Attr("entries: int = 0")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      ctx->set_output(2, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Edge bias grad for Convolution.
)doc");

template <typename T, typename V>
class EdgeBiasGradOp : public OpKernel {
 public:
  explicit EdgeBiasGradOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("layout",  &layout_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("entries", &entries_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",   &bench_  ));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy  = ctx->input(0);
    const Tensor& x   = ctx->input(1);
    const Tensor& g   = ctx->input(2);
    const Tensor& lut = ctx->input(3);

    uint rank = dy.dims();
    uint N    = dy.dim_size(0);
    uint MPQ  = 1, K, edges;
    // NCHW
    if (layout_ == 0)
    {
      K = dy.dim_size(1);
      for (int i = 2; i < rank; i++)
        MPQ *= dy.dim_size(i);

      edges = g.dim_size(1);
    }
    // NHWC
    else
    {
      K = dy.dim_size(rank-1);
      for (int i = 1; i < rank-1; i++)
        MPQ *= dy.dim_size(i);

      edges = g.dim_size(0);
    }
    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    // in place
    ctx->set_output(0, dy);

    Tensor* dg = nullptr;
    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &dg));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, g.shape(), &db));

              V*  dy_ptr = (V*)dy.flat<T>().data();
          float*  dg_ptr = dg->flat<float>().data();
          float*  db_ptr = db->flat<float>().data();
    const     V*   x_ptr = (const V*)x.flat<T>().data();
    const float*   g_ptr = g.flat<float>().data();
    const   int* lut_ptr = lut.flat<int32>().data();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "EdgeBiasGrad N:%3d,K:%3d,E:%2d L:%d", N, K, edges, layout_);
      bench = new Benchmark(stream, bench_string, 3*N*K*entries_*sizeof(V) + 3*K*edges*sizeof(float), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EdgeBiasBackward<V>(stream, dy_ptr, dg_ptr, db_ptr, x_ptr, g_ptr, lut_ptr, edges, MPQ, K, N, layout_);

    if (bench) delete bench;
  }
 private:
  int layout_, bench_, entries_;
};

REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"),EdgeBiasGradOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EdgeBiasGradOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EdgeBiasGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EdgeBiasGradOp<BHALF,bhalf>);




