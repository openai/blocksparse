
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


REGISTER_OP("EmbeddingLookup")
    .Input("emb: T")
    .Input("idx: TI")
    .Input("c: int32") // needed for gradient pass
    .Output("y: T")
    .Attr("T:  {float, half, bfloat16}")
    .Attr("TI: {int32, uint16, uint8}")
    .Attr("sorted: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx) {

      ShapeHandle emb = ctx->input(0);
      ShapeHandle idx = ctx->input(1);
      if (ctx->RankKnown(emb) && ctx->RankKnown(idx))
      {
        int rank = ctx->Rank(idx);

        std::vector<DimensionHandle> dims;
        dims.reserve(rank+1);
        for (int i = 0; i < rank; ++i)
            dims.emplace_back(ctx->Dim(idx, i));
        dims.emplace_back(ctx->Dim(emb, 1));

        ctx->set_output(0, ctx->MakeShape(dims));
      }
      else
        ctx->set_output(0, ctx->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
EmbeddingLookup.
)doc");

template <typename TI, typename T> bool EmbeddingLookup(CUstream stream, int SMs, T* y, const TI* idx, const T* w, int nIdx, int C, int K);

template <typename TI, typename T, typename V>
class EmbeddingLookupOp : public OpKernel {
 public:
  explicit EmbeddingLookupOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bench", &bench_));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor& emb = ctx->input(0);
    const Tensor& idx = ctx->input(1);

    OP_REQUIRES(ctx, emb.dim_size(0) == ctx->input(2).scalar<int32>()(), errors::InvalidArgument("Bad emb channels arg"));

    int C    = emb.dim_size(0);
    int K    = emb.dim_size(1);
    int rank = idx.dims();
    int nIdx = 1;
    TensorShape shape;
    for (int i = 0; i < rank; ++i)
    {
      int dim = idx.dim_size(i);
      nIdx *= dim;
      shape.AddDim(dim);
    }
    shape.AddDim(K);

    Tensor* y = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &y));

           V*   y_ptr = (V*)y->flat<T>().data();
    const  V* emb_ptr = (const V*)emb.flat<T>().data();
    const TI* idx_ptr = idx.flat<TI>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "EmbeddingLookup     nIdx:%7d, C:%5d, K:%4d", nIdx, C, K);
      bench = new Benchmark(stream, bench_string, nIdx*sizeof(TI) + 2*nIdx*K*sizeof(V), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EmbeddingLookup<TI,V>(stream, SMs_, y_ptr, idx_ptr, emb_ptr, nIdx, C, K);

    if (bench) delete bench;
  }
  int SMs_, bench_;
};
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupOp<int32,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupOp<int32,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupOp<int32,BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupOp<uint16,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupOp<uint16,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupOp<uint16,BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupOp<uint8,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupOp<uint8,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookup").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupOp<uint8,BHALF,bhalf>);


REGISTER_OP("EmbeddingLookupGrad")
    .Input("dy: T")
    .Input("idx: TI")
    .Input("c: int32")
    .Output("dw: float")
    .Attr("T:  {float, half, bfloat16}")
    .Attr("TI: {int32, uint16, uint8}")
    .Attr("sorted: bool = true")
    .Attr("bench: int = 0")
    .SetShapeFn([](InferenceContext* ctx)
    {
      ShapeHandle dy = ctx->input(0);
      if (ctx->RankKnown(dy))
      {
        DimensionHandle c_dim;
        TF_RETURN_IF_ERROR(ctx->MakeDimForScalarInput(2, &c_dim));

        ctx->set_output(0, ctx->MakeShape({ c_dim, ctx->Dim(dy, ctx->Rank(dy)-1) }));
      }
      else
        ctx->set_output(0, ctx->UnknownShape());

      return Status::OK();
    })
    .Doc(R"doc(
EmbeddingLookupGrad.
)doc");

template <typename TI, typename TG> bool EmbeddingLookupGrad(CUstream stream, int SMs, float* dw, const TI* idx, const TG* dy, int nIdx, int C, int K, bool sorted);

template <typename TI, typename T, typename V>
class EmbeddingLookupGradOp : public OpKernel {
 public:
  explicit EmbeddingLookupGradOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sorted", &sorted_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr( "bench", &bench_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor&  dy = ctx->input(0);
    const Tensor& idx = ctx->input(1);

    int    C = ctx->input(2).scalar<int32>()();
    int    K = dy.dim_size(dy.dims()-1);
    int nIdx = idx.shape().num_elements();
    TensorShape shape({ C, K });

    Tensor* dw = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &dw));

       float*  dw_ptr = dw->flat<float>().data();
    const  V*  dy_ptr = (const V*)dy.flat<T>().data();
    const TI* idx_ptr = idx.flat<TI>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Benchmark* bench = nullptr;
    if (bench_)
    {
      char bench_string[256];
      sprintf(bench_string, "EmbeddingLookupGrad nIdx:%7d, C:%5d, K:%4d, S:%d", nIdx, C, K, sorted_);
      bench = new Benchmark(stream, bench_string, C*K*sizeof(float) + nIdx*sizeof(TI) + nIdx*K*(sizeof(V) + sizeof(float)), 0, bench_);
    }

    int repeat = bench_ ? bench_ : 1;
    for (int i = 0; i < repeat; i++)
      EmbeddingLookupGrad<TI,V>(stream, SMs_, dw_ptr, idx_ptr, dy_ptr, nIdx, C, K, sorted_);

    if (bench) delete bench;
  }
  int SMs_, bench_;
  bool sorted_;
};
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupGradOp<int32,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<int32,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<int32>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<int32,BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupGradOp<uint16,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<uint16,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint16>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<uint16,BHALF,bhalf>);

REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<FLOAT>("T").HostMemory("c"),EmbeddingLookupGradOp<uint8,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<EHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<uint8,EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupGrad").Device(DEVICE_GPU).TypeConstraint<uint8>("TI").TypeConstraint<BHALF>("T").HostMemory("c"),EmbeddingLookupGradOp<uint8,BHALF,bhalf>);
