
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

template <typename T, typename V>
bool Adafactor(CUstream stream, uint SMs, float* cv, float* rv, float* x, float* means, float* param, const T* grad, const float* norm_scale, float grad_scale, float learning_rate, float decay, float epsilon, float clip_thresh, uint C, uint K, float saturate, bool zero_infs, bool zero_nans);

REGISTER_OP("Adafactor2d")
    .Input("param: Ref(float)")
    .Input("cv: Ref(float)")
    .Input("rv: Ref(float)")
    .Input("grad: T")
    .Input("decay: float")         // scalar host tensor
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_thresh: float")   // scalar host tensor
    .Input("norm_scale: n_norm * float")
    .Output("out_param: Ref(float)")
    .Output("out_cv: Ref(float)")
    .Output("out_rv: Ref(float)")
    .Output("temp_x: float")
    .Output("temp_m: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 1e-30")
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("n_norm: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      c->set_output(3, c->input(3));
      c->set_output(4, c->MakeShape({ c->MakeDim(2) }) );
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Adafactor optimizer on 2d tensors.
    )doc");

template <typename T, typename V1, typename V4>
class Adafactor2dOp : public OpKernel {
 public:
  explicit Adafactor2dOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",   &epsilon_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",  &saturate_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs", &zero_infs_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans", &zero_nans_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->forward_ref_input_to_ref_output(1, 1);
    ctx->forward_ref_input_to_ref_output(2, 2);

    const Tensor& grad  = ctx->input(3);
    const Tensor& decay = ctx->input(4);
    const Tensor& lr    = ctx->input(5);
    const Tensor& scal  = ctx->input(6);
    const Tensor& clip  = ctx->input(7);

    OpInputList norm_scale;
    ctx->input_list("norm_scale", &norm_scale);
    const float* norm_scale_ptr = norm_scale.size() > 0 ? norm_scale[0].flat<float>().data() : NULL;

    Tensor param = ctx->mutable_input(0, false);
    Tensor cv    = ctx->mutable_input(1, true);
    Tensor rv    = ctx->mutable_input(2, false);

    OP_REQUIRES(ctx, param.dims() == 2, errors::InvalidArgument("only applies to 2d params"));

    uint C = param.dim_size(0);
    uint K = param.dim_size(1);

    OP_REQUIRES(ctx, cv.shape().num_elements() == K, errors::InvalidArgument("bad cv shape"));
    OP_REQUIRES(ctx, rv.shape().num_elements() == C, errors::InvalidArgument("bad rv shape"));

    Tensor* x; OP_REQUIRES_OK(ctx, ctx->allocate_output(3,  param.shape(),      &x));
    Tensor* m; OP_REQUIRES_OK(ctx, ctx->allocate_output(4,  TensorShape({ 2 }), &m));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Adafactor<V1,V4>(stream, SMs_,
      cv.flat<float>().data(),
      rv.flat<float>().data(),
      x->flat<float>().data(),
      m->flat<float>().data(),
      param.flat<float>().data(),
      (const V1*)grad.flat<T>().data(),
      norm_scale_ptr,
      scal.scalar<float>()(), lr.scalar<float>()(), decay.scalar<float>()(), epsilon_, clip.scalar<float>()(), C, K, saturate_, zero_infs_, zero_nans_
    );
  }
  uint SMs_;
  float epsilon_, saturate_;
  bool zero_infs_, zero_nans_;
};
REGISTER_KERNEL_BUILDER(Name("Adafactor2d").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor2dOp<FLOAT,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Adafactor2d").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor2dOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Adafactor2d").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor2dOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("Adafactor1d")
    .Input("param: Ref(float)")
    .Input("cv: Ref(float)")
    .Input("grad: T")
    .Input("decay: float")         // scalar host tensor
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_thresh: float")   // scalar host tensor
    .Input("norm_scale: n_norm * float")
    .Output("out_param: Ref(float)")
    .Output("out_cv: Ref(float)")
    .Output("temp_x: float")
    .Output("temp_m: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 1e-30")
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("n_norm: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      c->set_output(3, c->MakeShape({ c->MakeDim(2) }) );
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Adafactor optimizer on 1d tensors.
    )doc");

template <typename T, typename V1, typename V4>
class Adafactor1dOp : public OpKernel {
 public:
  explicit Adafactor1dOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",   &epsilon_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",  &saturate_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs", &zero_infs_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans", &zero_nans_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->forward_ref_input_to_ref_output(1, 1);

    const Tensor& grad  = ctx->input(2);
    const Tensor& decay = ctx->input(3);
    const Tensor& lr    = ctx->input(4);
    const Tensor& scal  = ctx->input(5);
    const Tensor& clip  = ctx->input(6);

    OpInputList norm_scale;
    ctx->input_list("norm_scale", &norm_scale);
    const float* norm_scale_ptr = norm_scale.size() > 0 ? norm_scale[0].flat<float>().data() : NULL;

    Tensor param = ctx->mutable_input(0, false);
    Tensor cv    = ctx->mutable_input(1, false);

    OP_REQUIRES(ctx, param.dims() == 1 || (param.dims() == 2 && param.dim_size(0) == 1), errors::InvalidArgument("only applies to 1d params"));

    uint C = 1;
    uint K = param.shape().num_elements();

    OP_REQUIRES(ctx, cv.shape().num_elements() == K, errors::InvalidArgument("bad cv shape"));

    Tensor* x; OP_REQUIRES_OK(ctx, ctx->allocate_output(2,  param.shape(),      &x));
    Tensor* m; OP_REQUIRES_OK(ctx, ctx->allocate_output(3,  TensorShape({ 2 }), &m));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    Adafactor<V1,V4>(stream, SMs_,
      cv.flat<float>().data(),
      NULL,
      x->flat<float>().data(),
      m->flat<float>().data(),
      param.flat<float>().data(),
      (const V1*)grad.flat<T>().data(),
      norm_scale_ptr,
      scal.scalar<float>()(), lr.scalar<float>()(), decay.scalar<float>()(), epsilon_, clip.scalar<float>()(), C, K, saturate_, zero_infs_, zero_nans_
    );
  }
  uint SMs_;
  float epsilon_, saturate_;
  bool zero_infs_, zero_nans_;
};
REGISTER_KERNEL_BUILDER(Name("Adafactor1d").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor1dOp<FLOAT,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Adafactor1d").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor1dOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Adafactor1d").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("decay").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_thresh"), Adafactor1dOp<BHALF,bhalf,bhalf4>);



template <typename TG, typename TR> bool ApplyAdam(CUstream stream, uint SMs, const TG* grad, const float* norm_scale, float* param, TR* mean, TR* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_emb, float saturate, bool zero_infs, bool zero_nans);

REGISTER_OP("Adam")
    .Input("grad: TG")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_sigma: float")    // scalar host tensor
    .Input("norm_scale: n_norm * float")
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var: float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("lazy_emb: bool = false")
    .Attr("n_norm: int >= 0")
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

    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_mean", &decay_mean_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_var",  &decay_var_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",    &epsilon_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lazy_emb",   &lazy_emb_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",   &saturate_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs",  &zero_infs_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans",  &zero_nans_  ));
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

    OpInputList norm_scale;
    ctx->input_list("norm_scale", &norm_scale);
    const float* norm_scale_ptr = norm_scale.size() > 0 ? norm_scale[0].flat<float>().data() : NULL;

    Tensor param = ctx->mutable_input(1, false);
    Tensor mean  = ctx->mutable_input(2, false);
    Tensor var   = ctx->mutable_input(3, false);

    uint size, K;
    if (lazy_emb_)
    {
      OP_REQUIRES(ctx, param.dims() == 2, errors::InvalidArgument("lazy_emb only applies to 2d embedding params"));
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
      norm_scale_ptr,
      param.flat<float>().data(),
      (VR*)mean.flat<TR>().data(),
      (VR*)var.flat<TR>().data(),
      lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), size, K, saturate_, zero_infs_, zero_nans_
    );
  }
  uint SMs_;
  bool zero_infs_, zero_nans_, lazy_emb_;
  float decay_mean_, decay_var_, epsilon_, saturate_;
};
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<FLOAT,float,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<EHALF,ehalf,FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<BHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<BHALF,bhalf,FLOAT,float>);


template <typename TG, typename TR> bool ApplyAdamGated(CUstream stream, const float* gate, const TG* grad, const float* norm_scale, float* param, TR* mean, TR* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);

REGISTER_OP("AdamGated")
    .Input("gate: float")
    .Input("grad: TG")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_sigma: float")    // scalar host tensor
    .Input("norm_scale: n_norm * float")
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var: float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("n_norm: int >= 0")
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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",   &saturate_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs",  &zero_infs_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans",  &zero_nans_ ));
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

    OpInputList norm_scale;
    ctx->input_list("norm_scale", &norm_scale);
    const float* norm_scale_ptr = norm_scale.size() > 0 ? norm_scale[0].flat<float>().data() : NULL;

    Tensor param = ctx->mutable_input(2, false);
    Tensor mean  = ctx->mutable_input(3, false);
    Tensor var   = ctx->mutable_input(4, false);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    ApplyAdamGated<VG,VR>(stream,
      gate.flat<float>().data(),
      (const VG*)grad.flat<TG>().data(),
      norm_scale_ptr,
      param.flat<float>().data(),
      (VR*)mean.flat<TR>().data(),
      (VR*)var.flat<TR>().data(),
      lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), blocks, bsize, saturate_, zero_infs_, zero_nans_
    );
  }
  float decay_mean_, decay_var_, epsilon_, saturate_;
  bool zero_infs_, zero_nans_;
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


template <typename T, typename V> bool ReduceSumSquared(CUstream stream, uint SMs, float* sum_squared, const T* x, uint size, float grad_scale, float saturate, bool zero_infs, bool zero_nans, uint tensor_idx, uint tensor_cnt);
bool ComputeClipNorm(CUstream stream, float* l2norm, float* scale, float* sum_squared, float clip_norm, uint tensor_cnt);

REGISTER_OP("ClipGlobalNorm")
    .Input("grad_scale: float") // scalar host tensor
    .Input("clip_norm:  float") // scalar host tensor
    .Input("x_float: n_float * float")
    .Input("x_ehalf: n_ehalf * half")
    .Input("x_bhalf: n_bhalf * bfloat16")
    .Output("l2norm: float")
    .Output("scale:  float")
    .Output("temp:   float")
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("n_float: int >= 0")
    .Attr("n_ehalf: int >= 0")
    .Attr("n_bhalf: int >= 0")
    .SetShapeFn([](InferenceContext* ctx)
    {
      ctx->set_output(0, ctx->MakeShape({ ctx->MakeDim(1) }));
      ctx->set_output(1, ctx->MakeShape({ ctx->MakeDim(1) }));
      return Status::OK();
    })
    .Doc(R"doc(
ReduceSumSquared, scalar float output.  Used in computing norms.
)doc");

class ClipGlobalNormOp : public OpKernel
{
 public:
  explicit ClipGlobalNormOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",  &saturate_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs", &zero_infs_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans", &zero_nans_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    float grad_scale = ctx->input(0).scalar<float>()();
    float clip_norm  = ctx->input(1).scalar<float>()();

    OpInputList x_float, x_ehalf, x_bhalf;
    ctx->input_list("x_float", &x_float);
    ctx->input_list("x_ehalf", &x_ehalf);
    ctx->input_list("x_bhalf", &x_bhalf);

    uint tensor_idx = 0;
    uint tensor_cnt = x_float.size() + x_ehalf.size() + x_bhalf.size();

    Tensor *l2norm, *scale, *sum_sqr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0,  TensorShape({ 1 }), &l2norm));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  TensorShape({ 1 }), &scale));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2,  TensorShape({ tensor_cnt }), &sum_sqr));

    float* l2norm_ptr  = l2norm->flat<float>().data();
    float* scale_ptr   = scale->flat<float>().data();
    float* sum_sqr_ptr = sum_sqr->flat<float>().data();

    for (int i = 0; i < x_float.size(); i++)
    {
      uint size = x_float[i].shape().num_elements();
      const float* x_ptr = (const float*)x_float[i].flat<FLOAT>().data();

      ReduceSumSquared<float,float4>(stream, SMs_, sum_sqr_ptr, x_ptr, size, grad_scale, saturate_, zero_infs_, zero_nans_, tensor_idx, tensor_cnt);
      tensor_idx++;
    }
    for (int i = 0; i < x_ehalf.size(); i++)
    {
      uint size = x_ehalf[i].shape().num_elements();
      const ehalf* x_ptr = (const ehalf*)x_ehalf[i].flat<EHALF>().data();

      ReduceSumSquared<ehalf,ehalf4>(stream, SMs_, sum_sqr_ptr, x_ptr, size, grad_scale, saturate_, zero_infs_, zero_nans_, tensor_idx, tensor_cnt);
      tensor_idx++;
    }
    for (int i = 0; i < x_bhalf.size(); i++)
    {
      uint size = x_bhalf[i].shape().num_elements();
      const bhalf* x_ptr = (const bhalf*)x_bhalf[i].flat<BHALF>().data();

      ReduceSumSquared<bhalf,bhalf4>(stream, SMs_, sum_sqr_ptr, x_ptr, size, grad_scale, saturate_, zero_infs_, zero_nans_, tensor_idx, tensor_cnt);
      tensor_idx++;
    }

    ComputeClipNorm(stream, l2norm_ptr, scale_ptr, sum_sqr_ptr, clip_norm, tensor_cnt);
  }
  float saturate_;
  bool zero_infs_, zero_nans_;
  uint SMs_;
};
REGISTER_KERNEL_BUILDER(Name("ClipGlobalNorm").Device(DEVICE_GPU).HostMemory("grad_scale").HostMemory("clip_norm"), ClipGlobalNormOp);
