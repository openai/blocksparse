
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


template <typename TG, typename TR> bool ApplyAdam(CUstream stream, uint SMs, const TG* grad, float* param, TR* mean, TR* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_update, bool zero_nans);

REGISTER_OP("Adam")
    .Input("grad: TG")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_sigma: float")    // scalar host tensor
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var: float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("zero_nans: bool = false")
    .Attr("lazy_update: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      c->set_output(1, c->input(2));
      c->set_output(2, c->input(3));
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Adam optimizer
    )doc");

template <typename TG, typename VG, typename TR, typename VR>
class AdamOp : public OpKernel {
 public:
  explicit AdamOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_mean",  &decay_mean_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_var",   &decay_var_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",     &epsilon_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans",   &zero_nans_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lazy_update", &lazy_update_));
  }

  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    ctx->forward_ref_input_to_ref_output(1, 0);
    ctx->forward_ref_input_to_ref_output(2, 1);
    ctx->forward_ref_input_to_ref_output(3, 2);

    const Tensor& grad = ctx->input(0);
    const Tensor& lr   = ctx->input(4);
    const Tensor& scal = ctx->input(5);
    const Tensor& clip = ctx->input(6);

    Tensor param = ctx->mutable_input(1, false);
    Tensor mean  = ctx->mutable_input(2, false);
    Tensor var   = ctx->mutable_input(3, false);

    uint size, K;
    if (lazy_update_)
    {
      OP_REQUIRES(ctx, param.dims() == 2, errors::InvalidArgument("lazy_update only applies to 2d embedding params"));
      size = param.dim_size(0);
      K    = param.dim_size(1);
    }
    else
    {
      size = param.shape().num_elements();
      K = 0;
    }
    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    ApplyAdam<VG,VR>(stream, SMs_,
      (const VG*)grad.flat<TG>().data(),
      param.flat<float>().data(),
      (VR*)mean.flat<TR>().data(),
      (VR*)var.flat<TR>().data(),
      lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), size, K, zero_nans_
    );
  }
  uint SMs_;
  bool zero_nans_, lazy_update_;
  float decay_mean_, decay_var_, epsilon_;
};
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<FLOAT,float,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<EHALF,ehalf,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<BHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<BHALF,bhalf,FLOAT,float>);


template <typename TG, typename TR> bool ApplyAdamGated(CUstream stream, const float* gate, const TG* grad, float* param, TR* mean, TR* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize);

REGISTER_OP("AdamGated")
    .Input("gate: float")
    .Input("grad: TG")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_sigma: float")    // scalar host tensor
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var: float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Adam optimizer but skipping over blocks with zero gate values.
    )doc");

template <typename TG, typename VG, typename TR, typename VR>
class AdamGatedOp : public OpKernel {
 public:
  explicit AdamGatedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_mean", &decay_mean_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_var",  &decay_var_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",    &epsilon_   ));
  }

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(2, 0);
    ctx->forward_ref_input_to_ref_output(3, 1);
    ctx->forward_ref_input_to_ref_output(4, 2);

    const Tensor& gate = ctx->input(0);
    const Tensor& grad = ctx->input(1);
    const Tensor& lr   = ctx->input(5);
    const Tensor& scal = ctx->input(6);
    const Tensor& clip = ctx->input(7);


    Tensor param = ctx->mutable_input(2, false);
    Tensor mean  = ctx->mutable_input(3, false);
    Tensor var   = ctx->mutable_input(4, false);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    ApplyAdamGated<VG,VR>(stream,
      gate.flat<float>().data(),
      (const VG*)grad.flat<TG>().data(),
      param.flat<float>().data(),
      (VR*)mean.flat<TR>().data(),
      (VR*)var.flat<TR>().data(),
      lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), blocks, bsize
    );
  }
  float decay_mean_, decay_var_, epsilon_;
};
REGISTER_KERNEL_BUILDER(Name("AdamGated").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamGatedOp<FLOAT,float,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("AdamGated").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamGatedOp<EHALF,ehalf,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("AdamGated").Device(DEVICE_GPU).TypeConstraint<BHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamGatedOp<BHALF,bhalf,FLOAT,float>);


template <typename T> bool ApplyEma(     CUstream stream, T* ema, const T* param, float decay, uint size);
template <typename T> bool ApplyEmaGated(CUstream stream, T* ema, const T* param, const float* gate, float decay, uint blocks, uint bsize);

REGISTER_OP("Ema")
    .Input("ema: Ref(T)")
    .Input("param: T")
    .Output("out_ema: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("decay: float = 0.999")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Apply Exponential Moving Average
    )doc");

template <typename T, typename V>
class EmaOp : public OpKernel {
 public:
  explicit EmaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay", &decay_));
  }

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor  ema   = ctx->mutable_input(0, false);
    const Tensor& param = ctx->input(1);

    uint size = param.shape().num_elements();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    ApplyEma<V>(stream,
      (V*)ema.flat<T>().data(),
      (const V*)param.flat<T>().data(),
      decay_, size
    );
  }
  float decay_;
};

REGISTER_KERNEL_BUILDER(Name("Ema").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), EmaOp<FLOAT,float>);


REGISTER_OP("EmaGated")
    .Input("ema: Ref(T)")
    .Input("param: T")
    .Input("gate: float")
    .Output("out_ema: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("decay: float = 0.999")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Apply Exponential Moving Average
    )doc");

template <typename T, typename V>
class EmaGatedOp : public OpKernel {
 public:
  explicit EmaGatedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay", &decay_));
  }

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor  ema   = ctx->mutable_input(0, false);
    const Tensor& param = ctx->input(1);
    const Tensor& gate  = ctx->input(2);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    ApplyEmaGated<V>(stream,
      (V*)ema.flat<T>().data(),
      (const V*)param.flat<T>().data(),
      gate.flat<float>().data(),
      decay_, blocks, bsize
    );
  }
  float decay_;
};

REGISTER_KERNEL_BUILDER(Name("EmaGated").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), EmaGatedOp<FLOAT,float>);


template <typename T> bool BlocksparseL2Decay(CUstream stream, T* param, const float* gate, float rate, float epsilon, uint blocks, uint bsize);
template <typename T> bool BlocksparseMaxnormPrune(CUstream stream, const T* param, float* gate, float threshold, uint blocks, uint bsize);


REGISTER_OP("BlocksparseL2Decay")
    .Input("param: Ref(T)")
    .Input("rate: float") // scalar host tensor
    .Output("out_param: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Apply L2 regularization to blocksparse weights
    )doc");

template <typename T, typename V>
class BlocksparseL2DecayOp : public OpKernel {
 public:
  explicit BlocksparseL2DecayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor param = ctx->mutable_input(0, false);
    const Tensor& rate = ctx->input(1);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseL2Decay<V>(stream,
      (V*)param.flat<T>().data(),
      NULL,
      rate.scalar<float>()(),
      epsilon_, blocks, bsize
    );
  }
  float epsilon_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseL2Decay").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("rate"), BlocksparseL2DecayOp<FLOAT,float>);



REGISTER_OP("BlocksparseL2DecayGated")
    .Input("param: Ref(T)")
    .Input("gate: float")
    .Input("rate: float") // scalar host tensor
    .Output("out_param: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Apply L2 regularization to blocksparse weights, filtered by gate.
    )doc");

template <typename T, typename V>
class BlocksparseL2DecayGatedOp : public OpKernel {
 public:
  explicit BlocksparseL2DecayGatedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor  param = ctx->mutable_input(0, false);
    const Tensor& gate  = ctx->input(1);
    const Tensor& rate  = ctx->input(2);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseL2Decay<V>(stream,
      (V*)param.flat<T>().data(),
      gate.flat<float>().data(),
      rate.scalar<float>()(), epsilon_, blocks, bsize
    );
  }
  float epsilon_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseL2DecayGated").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("rate"), BlocksparseL2DecayGatedOp<FLOAT,float>);



REGISTER_OP("BlocksparseMaxnormPrune")
    .Input("gate: Ref(float)")
    .Input("param: T")
    .Input("threshold: float") // scalar host tensor
    .Output("out_gate: Ref(float)")
    .Attr("T: {float, half, bfloat16}")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Set gate to zero for each block where maxnorm falls below threshold
    )doc");

template <typename T, typename V>
class BlocksparseMaxnormPruneOp : public OpKernel {
 public:
  explicit BlocksparseMaxnormPruneOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {

    ctx->forward_ref_input_to_ref_output(0, 0);

           Tensor gate   = ctx->mutable_input(0, false);
    const Tensor& param  = ctx->input(1);
    const Tensor& thresh = ctx->input(2);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseMaxnormPrune<V>(stream,
      (const V*)param.flat<T>().data(),
      gate.flat<float>().data(),
      thresh.scalar<float>()(),
      blocks, bsize
    );
  }
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseMaxnormPrune").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("threshold"), BlocksparseMaxnormPruneOp<FLOAT,float>);

