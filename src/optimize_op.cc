
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

template <typename TG, typename TRM, typename TRV> bool BlocksparseAdam(CUstream stream,
  float* param, TRM* mean, TRV* var,
  const TG* grad,
  const float* lr_select,
  const float* norm_scale,
  float lr_old, float lr_new,
  float decay_mean, float decay_var, float epsilon,
  float grad_scale, float saturate, bool zero_infs, bool zero_nans,
  uint blocks, uint bsize);

REGISTER_OP("BlocksparseAdam")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("grad: TG")
    .Input("grad_scale: float") // scalar host tensor
    .Input("lr_old: float")     // scalar host tensor
    .Input("lr_new: float")     // scalar host tensor
    .Input("lr_select: n_select * float")
    .Input("norm_scale: n_norm * float")
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var:  float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("n_norm: int >= 0")
    .Attr("n_select: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Blocksparse Adam optimizer
    )doc");

template <typename TG, typename VG, typename TR, typename VRM, typename VRV>
class BlocksparseAdamOp : public OpKernel {
 public:
  explicit BlocksparseAdamOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_mean", &decay_mean_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_var",  &decay_var_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon",    &epsilon_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("saturate",   &saturate_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_infs",  &zero_infs_  ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_nans",  &zero_nans_  ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    ctx->forward_ref_input_to_ref_output(0, 0);
    ctx->forward_ref_input_to_ref_output(1, 1);
    ctx->forward_ref_input_to_ref_output(2, 2);

    Tensor param = ctx->mutable_input(0, false);
    Tensor mean  = ctx->mutable_input(1, false);
    Tensor var   = ctx->mutable_input(2, false);

    const Tensor& grad       = ctx->input(3);
    const Tensor& grad_scale = ctx->input(4);
    const Tensor& lr_old     = ctx->input(5);
    const Tensor& lr_new     = ctx->input(6);

    OpInputList norm_scale, lr_select;
    ctx->input_list("lr_select",  &lr_select);
    ctx->input_list("norm_scale", &norm_scale);
    const float* lr_select_ptr  = lr_select.size()  > 0 ? lr_select[0].flat<float>().data()  : NULL;
    const float* norm_scale_ptr = norm_scale.size() > 0 ? norm_scale[0].flat<float>().data() : NULL;

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    BlocksparseAdam<VG,VRM,VRV>(stream,
      param.flat<float>().data(),
      (VRM*)mean.flat<TR>().data(),
      (VRV*)var.flat<TR>().data(),
      (const VG*)grad.flat<TG>().data(),
      lr_select_ptr,
      norm_scale_ptr,
      lr_old.scalar<float>()(),
      lr_new.scalar<float>()(),
      decay_mean_, decay_var_, epsilon_,
      grad_scale.scalar<float>()(), saturate_, zero_infs_, zero_nans_,
      blocks, bsize);
  }
  bool zero_infs_, zero_nans_;
  float decay_mean_, decay_var_, epsilon_, saturate_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseAdam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<FLOAT>("TR").HostMemory("grad_scale").HostMemory("lr_old").HostMemory("lr_new"), BlocksparseAdamOp<FLOAT,float,FLOAT,float,float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseAdam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("grad_scale").HostMemory("lr_old").HostMemory("lr_new"), BlocksparseAdamOp<EHALF,ehalf,FLOAT,float,float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseAdam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<EHALF>("TR").HostMemory("grad_scale").HostMemory("lr_old").HostMemory("lr_new"), BlocksparseAdamOp<FLOAT,float,EHALF,mhalf,vhalf>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseAdam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<EHALF>("TR").HostMemory("grad_scale").HostMemory("lr_old").HostMemory("lr_new"), BlocksparseAdamOp<EHALF,ehalf,EHALF,mhalf,vhalf>);


template <typename TG, typename TRM, typename TRV> bool ApplyAdam(     CUstream stream, uint SMs,          const TG* grad, const float* norm_scale, float* param, TRM* mean, TRV* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint size, uint lazy_emb, float saturate, bool zero_infs, bool zero_nans);
template <typename TG, typename TRM, typename TRV> bool ApplyAdamGated(CUstream stream, const float* gate, const TG* grad, const float* norm_scale, float* param, TRM* mean, TRV* var, float lr, float decay_mean, float decay_var, float epsilon, float grad_scale, float clip_sigma, uint blocks, uint bsize, float saturate, bool zero_infs, bool zero_nans);

REGISTER_OP("Adam")
    .Input("grad: TG")
    .Input("param: Ref(float)")
    .Input("mean: Ref(TR)")
    .Input("var: Ref(TR)")
    .Input("learning_rate: float") // scalar host tensor
    .Input("grad_scale: float")    // scalar host tensor
    .Input("clip_sigma: float")    // scalar host tensor
    .Input("norm_scale: n_norm * float")
    .Input("gate: n_gate * float")
    .Output("out_param: Ref(float)")
    .Output("out_mean: Ref(TR)")
    .Output("out_var: Ref(TR)")
    .Attr("TG: {float, half, bfloat16}")
    .Attr("TR: {float, half, bfloat16}")
    .Attr("decay_mean: float = 0.9")
    .Attr("decay_var:  float = 0.999")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("saturate: float = 0.0")
    .Attr("zero_infs: bool = false")
    .Attr("zero_nans: bool = false")
    .Attr("lazy_emb: bool = false")
    .Attr("n_norm: int >= 0")
    .Attr("n_gate: int >= 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(1));
      c->set_output(1, c->input(2));
      c->set_output(2, c->input(3));
      return Status::OK();
    })
    .Doc(R"doc(
Apply the Adam optimizer
    )doc");

template <typename TG, typename VG, typename TR, typename VRM, typename VRV>
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

    OpInputList norm_scale, gate;
    ctx->input_list("norm_scale", &norm_scale);
    ctx->input_list("gate", &gate);
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

    if (gate.size() > 0)
    {
      uint blocks = param.dim_size(0);
      uint bsize  = param.dim_size(1);

      ApplyAdamGated<VG,VRM,VRV>(stream,
        gate[0].flat<float>().data(),
        (const VG*)grad.flat<TG>().data(),
        norm_scale_ptr,
        param.flat<float>().data(),
        (VRM*)mean.flat<TR>().data(),
        (VRV*)var.flat<TR>().data(),
        lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), blocks, bsize, saturate_, zero_infs_, zero_nans_
      );
    }

    ApplyAdam<VG,VRM,VRV>(stream, SMs_,
      (const VG*)grad.flat<TG>().data(),
      norm_scale_ptr,
      param.flat<float>().data(),
      (VRM*)mean.flat<TR>().data(),
      (VRV*)var.flat<TR>().data(),
      lr.scalar<float>()(), decay_mean_, decay_var_, epsilon_, scal.scalar<float>()(), clip.scalar<float>()(), size, K, saturate_, zero_infs_, zero_nans_
    );
  }
  uint SMs_;
  bool zero_infs_, zero_nans_, lazy_emb_;
  float decay_mean_, decay_var_, epsilon_, saturate_;
};
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<FLOAT,float,FLOAT,float,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<EHALF,ehalf,FLOAT,float,float>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<BHALF>("TG").TypeConstraint<FLOAT>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<BHALF,bhalf,FLOAT,float,float>);

REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<FLOAT>("TG").TypeConstraint<EHALF>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<FLOAT,float,EHALF,mhalf,vhalf>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<EHALF>("TG").TypeConstraint<EHALF>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<EHALF,ehalf,EHALF,mhalf,vhalf>);
REGISTER_KERNEL_BUILDER(Name("Adam").Device(DEVICE_GPU).TypeConstraint<BHALF>("TG").TypeConstraint<EHALF>("TR").HostMemory("learning_rate").HostMemory("grad_scale").HostMemory("clip_sigma"), AdamOp<BHALF,bhalf,EHALF,mhalf,vhalf>);


template <typename T> bool ApplyEma(     CUstream stream, T* ema, const float* param, float decay, uint size);
template <typename T> bool ApplyEmaGated(CUstream stream, T* ema, const float* param, const float* gate, float decay, uint blocks, uint bsize);

REGISTER_OP("Ema")
    .Input("ema: Ref(T)")
    .Input("param: float")
    .Input("gate: n_gate * float")
    .Output("out_ema: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("decay: float = 0.999")
    .Attr("n_gate: int >= 0")
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
  explicit EmaOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("decay", &decay_));
  }

  void Compute(OpKernelContext* ctx) override
  {
    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor  ema   = ctx->mutable_input(0, false);
    const Tensor& param = ctx->input(1);

    OpInputList gate;
    ctx->input_list("gate", &gate);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    if (gate.size() > 0)
    {
      uint blocks = param.dim_size(0);
      uint bsize  = param.dim_size(1);

      ApplyEmaGated<V>(stream,
        (V*)ema.flat<T>().data(),
        param.flat<float>().data(),
        gate[0].flat<float>().data(),
        decay_, blocks, bsize
      );
    }
    else
    {
      uint size = param.shape().num_elements();

      ApplyEma<V>(stream,
        (V*)ema.flat<T>().data(),
        param.flat<float>().data(),
        decay_, size
      );
    }
  }
  float decay_;
};
REGISTER_KERNEL_BUILDER(Name("Ema").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), EmaOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Ema").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), EmaOp<EHALF,ehalf>);


template <typename T> bool BlocksparseL2Decay(CUstream stream, T* param, const float* gate, float rate, float epsilon, uint blocks, uint bsize);

REGISTER_OP("BlocksparseL2Decay")
    .Input("param: Ref(T)")
    .Input("rate: float") // scalar host tensor
    .Input("gate: n_gate * float")
    .Output("out_param: Ref(T)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("epsilon: float = 0.00000001") // 1e-8
    .Attr("n_gate: int >= 0")
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
  explicit BlocksparseL2DecayOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override
  {

    ctx->forward_ref_input_to_ref_output(0, 0);

          Tensor param = ctx->mutable_input(0, false);
    const Tensor& rate = ctx->input(1);

    OpInputList gate;
    ctx->input_list("gate", &gate);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseL2Decay<V>(stream,
      (V*)param.flat<T>().data(),
      gate.size() > 0 ? gate[0].flat<float>().data() : NULL,
      rate.scalar<float>()(), epsilon_, blocks, bsize
    );
  }
  float epsilon_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseL2Decay").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("rate"), BlocksparseL2DecayOp<FLOAT,float>);


template <typename T> bool BlocksparseNorm(CUstream stream, float* norm, const T* param, uint blocks, uint bsize, uint norm_type);

REGISTER_OP("BlocksparseNorm")
    .Input("param: T")
    .Output("norm: float")
    .Attr("T: {float, half, bfloat16}")
    .Attr("norm_type: int = 0")
    .SetShapeFn([](InferenceContext* ctx)
    {
      ShapeHandle p = ctx->input(0);
      if (ctx->RankKnown(p))
        ctx->set_output(0, ctx->Vector(ctx->Dim(p, 0)));
      else
        ctx->set_output(0, ctx->UnknownShape());

      return Status::OK();
    })
    .Doc(R"doc(
BlocksparseNorm
)doc");

template <typename T, typename V>
class BlocksparseNormOp : public OpKernel {
 public:
  explicit BlocksparseNormOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("norm_type",  &norm_type_));
  }

  void Compute(OpKernelContext* ctx) override
  {
    const Tensor& param = ctx->input(0);

    uint blocks = param.dim_size(0);
    uint bsize  = param.dim_size(1);

    Tensor *norm;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0,  TensorShape({ blocks }), &norm));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseNorm<V>(stream,
      norm->flat<float>().data(),
      (const V*)param.flat<T>().data(),
      blocks, bsize, norm_type_
    );
  }
  int32 norm_type_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseNorm").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), BlocksparseNormOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseNorm").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), BlocksparseNormOp<EHALF,ehalf>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseNorm").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), BlocksparseNormOp<BHALF,bhalf>);


bool BlocksparsePrune(CUstream stream, uint SMs, float* gate, const uint* idx, uint blocks, uint keep);

REGISTER_OP("BlocksparsePrune")
    .Input("gate: Ref(float)")
    .Input("idx: int32")
    .Input("sparsity: float") // scalar host tensor
    .Input("step: S")         // scalar host tensor
    .Output("out_gate: Ref(float)")
    .Attr("S: {int32, int64}")
    .Attr("frequency: int = 1")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
BlocksparsePrune
    )doc");

class BlocksparsePruneOp : public OpKernel {
 public:
  explicit BlocksparsePruneOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frequency",  &frequency_));
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    const Tensor& stp = ctx->input(3);
    int32 step = stp.dtype() == DT_INT64 ? (int32)stp.scalar<int64>()() : stp.scalar<int32>()();

    ctx->forward_ref_input_to_ref_output(0, 0);

    if (frequency_ > 0 && (frequency_ == 1 || (step % frequency_) == 0))
    {
            Tensor gate      = ctx->mutable_input(0, false);
      const Tensor& idx      = ctx->input(1);
      const Tensor& sparsity = ctx->input(2);

      float keep_frac = 1.0f - sparsity.scalar<float>()();

      // negative sparsity makes this a no-op
      if (keep_frac <= 1.0f)
      {
        uint blocks = gate.dim_size(0);
        uint keep   = (uint)((float)blocks * keep_frac + 0.5f);

        CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        BlocksparsePrune(stream, SMs_,
          gate.flat<float>().data(),
          (const uint*)idx.flat<int32>().data(),
          blocks, keep
        );
      }
    }
  }
  int32 frequency_;
  uint SMs_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparsePrune").Device(DEVICE_GPU).HostMemory("sparsity").HostMemory("step"), BlocksparsePruneOp);


template <typename T> bool BlocksparseThresholdPrune(CUstream stream, const T* param, float* gate, float threshold, uint blocks, uint bsize, uint norm_type);

REGISTER_OP("BlocksparseThresholdPrune")
    .Input("gate: Ref(float)")
    .Input("param: T")
    .Input("threshold: float") // scalar host tensor
    .Input("step: S")          // scalar host tensor
    .Output("out_gate: Ref(float)")
    .Attr("T: {float, half, bfloat16}")
    .Attr("S: {int32, int64}")
    .Attr("norm_type: int = 0")
    .Attr("frequency: int = 1")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Set gate to zero for each block where maxnorm falls below threshold
    )doc");

template <typename T, typename V>
class BlocksparseThresholdPruneOp : public OpKernel {
 public:
  explicit BlocksparseThresholdPruneOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("norm_type",  &norm_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frequency",  &frequency_));
  }

  void Compute(OpKernelContext* ctx) override
  {

    const Tensor& stp = ctx->input(3);
    int32 step = stp.dtype() == DT_INT64 ? (int32)stp.scalar<int64>()() : stp.scalar<int32>()();

    ctx->forward_ref_input_to_ref_output(0, 0);

    if (frequency_ > 0 && (frequency_ == 1 || (step % frequency_) == 0))
    {
            Tensor  gate   = ctx->mutable_input(0, false);
      const Tensor& param  = ctx->input(1);
      const Tensor& thresh = ctx->input(2);

      uint blocks = param.dim_size(0);
      uint bsize  = param.dim_size(1);

      CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

      BlocksparseThresholdPrune<V>(stream,
        (const V*)param.flat<T>().data(),
        gate.flat<float>().data(),
        thresh.scalar<float>()(),
        blocks, bsize, norm_type_
      );
    }
  }
  int norm_type_, frequency_;
};
REGISTER_KERNEL_BUILDER(Name("BlocksparseThresholdPrune").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("threshold").HostMemory("step"), BlocksparseThresholdPruneOp<FLOAT,float>);


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
      ctx->set_output(0, ctx->MakeShape({}));
      ctx->set_output(1, ctx->MakeShape({}));
      ctx->set_output(2, ctx->UnknownShape());
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
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0,  TensorShape({}), &l2norm));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  TensorShape({}), &scale));
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



REGISTER_OP("Recompute")
    .Input( "fwd: n_out * T")
    .Input( "bwd: n_out * T")
    .Output("out: n_out * T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("n_out: int >= 1")
    .SetShapeFn([](InferenceContext* ctx)
    {
      int n_out; TF_RETURN_IF_ERROR(ctx->GetAttr("n_out", &n_out));
      for (int i = 0; i < n_out; i++)
        ctx->set_output(i, ctx->input(i));
      return Status::OK();
    })
    .Doc(R"doc(
Recompute placeholder and passthrough node.
    )doc");

class RecomputeOp : public OpKernel {
 public:
  explicit RecomputeOp(OpKernelConstruction* ctx) : OpKernel(ctx)
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("n_out",  &n_out));
  }
  void Compute(OpKernelContext* ctx) override
  {
    for (int i = 0; i < n_out; i++)
      ctx->set_output(i, ctx->input(i));
  }
  int n_out;
};

REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_CPU).TypeConstraint<FLOAT>("T"), RecomputeOp);
REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_CPU).TypeConstraint<EHALF>("T"), RecomputeOp);
REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_CPU).TypeConstraint<BHALF>("T"), RecomputeOp);

REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T"), RecomputeOp);
REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"), RecomputeOp);
REGISTER_KERNEL_BUILDER(Name("Recompute").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"), RecomputeOp);
