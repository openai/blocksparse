
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


template <typename T> bool Quantize(CUstream stream, uint SMs, uint* entropy, T* y, const T* x, float round_scale, uint trunc_mask, float max_float, float min_float, uint exp_norm, uint size, int stochastic);
template <typename T> QuantStats QuantizationStats(CUstream stream, uint SMs, float* s, const T* x, float max_float, float ftz_float, uint size);

REGISTER_OP("Quantize")
    .Input("x: T")
    .Input("exp_max: int64")
    .Input("b_exp_max: int64")
    .Input("entropy: n_entropy * float")
    .Output("y: T")
    .Attr("T: {float, bfloat16}")
    .Attr("ebits: int")
    .Attr("fbits: int")
    .Attr("stoch: int = 0")
    .Attr("denorm: bool = true")
    .Attr("freq: int = 0")
    .Attr("freq2: int = 4")
    .Attr("mode: int = 0")
    .Attr("bias_pad: int = 2")
    .Attr("stdv_mul: float = 4.0")
    .Attr("logfile: string = ''")
    .Attr("b_ebits: int = 0")
    .Attr("b_fbits: int = 0")
    .Attr("b_stoch: int = 0")
    .Attr("b_denorm: bool = true")
    .Attr("b_freq: int = 0")
    .Attr("b_freq2: int = 4")
    .Attr("b_mode: int = 0")
    .Attr("b_bias_pad: int = 2")
    .Attr("b_stdv_mul: float = 4.0")
    .Attr("b_logfile: string = ''")
    .Attr("n_entropy: int >= 0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Quantize tensor to arbitrary precision.  Perform stochastic rounding if desired.
)doc");

template <typename T, typename V>
class QuantizeOp : public OpKernel {
 public:
  explicit QuantizeOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), count_(1), pow2_(1), pow2_count_(0), last_exp_(9999), max_stat_hi_(0.0f), max_stat_lo_(FLT_MAX) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ebits",    &ebits_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fbits",    &fbits_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stoch",    &stoch_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("denorm",   &denorm_   ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("freq",     &freq_     ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("freq2",    &freq2_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode",     &mode_     ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias_pad", &bias_pad_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stdv_mul", &stdv_mul_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("logfile",  &logfile_  ));

    // extra scaling for the random uint => float conversion
    // We need a value between [0, 2) that when shifted just below the ulp will provide our rounding add value.
    // Non stochastic rounding just uses the implied 1 in the mantissa giving us a .5 rounding add value.
    int stoch_bias = stoch_ ? 31 : 0;

    int rscale   = (127 - fbits_ - 1 - stoch_bias) << 23;
    round_scale_ = *(float*)&rscale;
    trunc_mask_  = 0xffffffff << (23 - fbits_);
    max_exp_     = (1 << ebits_) - 1;
    if (ebits_ == 8)
      max_exp_ -= 1; // fp32 reserves top bin for inf values
  }
  int UpdateExponent(int exp_max)
  {
    exp_max += 127;

    // constrain to exponents that fit inside floats
    if (exp_max < max_exp_)
        exp_max = max_exp_;

    int exp_min = exp_max - max_exp_ + 1 - (denorm_ ? fbits_: 0);

    if (exp_min < 2)
      exp_min = 2;

    int max_float = ((exp_max << 23) | 0x007fffff) & trunc_mask_;
    int min_float = exp_min << 23;
    int ftz_float = ((exp_min-1) << 23) | 0x00400000; // values below this are rounded to zero

    // amount to subtract off of exponent to bring smallest quantized subnormal value down to unbiased exponent of 1
    exp_norm_ = (exp_min - 1 - (denorm_ ? 0 : fbits_)) << 23;

    exp_max -= 127;

    max_float_ = *(float*)&max_float;
    min_float_ = *(float*)&min_float;
    ftz_float_ = *(float*)&ftz_float;
    last_exp_  = exp_max;
    return exp_max;
  }
  void Compute(OpKernelContext* ctx) override
  {
    if (SMs_ == 0)
      SMs_ = GetCountSMs();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    const Tensor&  x = ctx->input(0);
    const Tensor& em = ctx->input(1);
    OpInputList    e;  ctx->input_list("entropy", &e);

    Tensor* y; OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

          V*  y_ptr = (      V*)y->flat<T>().data();
    const V*  x_ptr = (const V*)x.flat<T>().data();
    uint*     e_ptr = e.size() > 0 ? (uint*)e[0].flat<float>().data() : NULL;
    int64* emax_ptr = (int64*)em.flat<int64>().data();

    uint size = x.NumElements();

    int emax  = (int)*emax_ptr;
    //printf("emax: %d\n", emax);
    if (emax != last_exp_)
      UpdateExponent(emax);

    if (freq_ && ((count_ & (pow2_-1)) == 0))
    {
      if ((pow2_ << 1) <= freq_)
      {
        if (pow2_count_ == freq2_)
        {
          pow2_     <<= 1;
          pow2_count_ = 0;
        }
        pow2_count_  += 1;
      }
      Tensor s; OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({sizeof(QuantStats)/4}), &s));
      float* stats_ptr = s.flat<float>().data();

      QuantStats stats = QuantizationStats<V>(stream, SMs_, stats_ptr, x_ptr, max_float_, ftz_float_, size);

      float mode_max = mode_ ? stats.mean + stats.stdv * stdv_mul_ : stats.max_val;

      if (stats.max_val < max_stat_lo_) max_stat_lo_ = stats.max_val;
      if (stats.max_val > max_stat_hi_) max_stat_hi_ = stats.max_val;

      *emax_ptr = UpdateExponent((*(int*)&mode_max >> 23)-127 + bias_pad_);

      if (!logfile_.empty())
      {
        if (FILE* log = fopen(logfile_.c_str(), "a"))
        {
          float mean_stdv5 = stats.mean + stats.stdv * 5.0f;

          fprintf(log, "%.3f\t%.3f\t%3d\t%3d\t%3d\t%3d\t%3d\t%3d\t%3d\t%3d\t%d\t%s\n",
            stats.sat_pct,
            stats.ftz_pct,
            (*(int*)&max_float_    >> 23)-127,
            (*(int*)&min_float_    >> 23)-127,
            (*(int*)&stats.max_val >> 23)-127,
            (*(int*)&stats.mean    >> 23)-127,
            (*(int*)&stats.stdv    >> 23)-127,
            (*(int*)&mean_stdv5    >> 23)-127,
            (*(int*)&max_stat_lo_  >> 23)-127,
            (*(int*)&max_stat_hi_  >> 23)-127,
            count_,
            this->name().c_str()
          );
          fclose(log);
        }
      }
    }

    Quantize<V>(stream, SMs_, e_ptr, y_ptr, x_ptr, round_scale_, trunc_mask_, max_float_, min_float_, exp_norm_, size, stoch_);

    count_ += 1;
  }
  uint  count_, trunc_mask_, exp_norm_, pow2_, pow2_count_;
  int   ebits_, fbits_, stoch_, freq_, freq2_, mode_, bias_pad_, SMs_, last_exp_, max_exp_;
  bool  denorm_;
  float stdv_mul_, round_scale_, max_float_, min_float_, ftz_float_, max_stat_hi_, max_stat_lo_;
  std::string logfile_;
};
REGISTER_KERNEL_BUILDER(Name("Quantize").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("exp_max").HostMemory("b_exp_max"),QuantizeOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("Quantize").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("exp_max").HostMemory("b_exp_max"),QuantizeOp<BHALF,bhalf>);



REGISTER_OP("LogStats")
    .Input("x: T")
    .Input("step: S")
    .Output("y: T")
    .Attr("T: {float, half, bfloat16}")
    .Attr("S: {int32, int64}")
    .Attr("sat_val: float")
    .Attr("ftz_val: float")
    .Attr("freq: int = 512")
    .Attr("bfreq: int = 512")
    .Attr("first_steps: list(int)")
    .Attr("logfile: string = ''")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Just collect and log basic stats on tensors. (mainly for fp16 tuning)
)doc");

template <typename T, typename V>
class LogStatsOp : public OpKernel {
 public:
  explicit LogStatsOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), max_stat_hi_(0.0f), max_stat_lo_(FLT_MAX), prev_step_(-1) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sat_val", &sat_val_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ftz_val", &ftz_val_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("freq",    &freq_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("logfile", &logfile_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("first_steps", &first_steps_ ));
  }
  void Compute(OpKernelContext* ctx) override
  {
    const Tensor&   x = ctx->input(0);
    const Tensor& stp = ctx->input(1);

    int step = stp.dtype() == DT_INT64 ? (int)stp.scalar<int64>()() : stp.scalar<int32>()();

    ctx->set_output(0, x);

    if (freq_)
    {
      bool test = false;
      if (step != prev_step_)
      {
        prev_step_ = step;
        if (step < freq_)
        {
          for (std::vector<int32>::iterator it = first_steps_.begin() ; it != first_steps_.end(); ++it)
            if (step == *it)
            {
              test = true;
              break;
            }
        }
        else
          test = (step & (freq_-1)) == 0;
      }

      if (test)
      {
        if (SMs_ == 0)
          SMs_ = GetCountSMs();

        const V* x_ptr = (const V*)x.flat<T>().data();
        uint size = x.NumElements();

        Tensor s; OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, TensorShape({sizeof(QuantStats)/4}), &s));
        float* stats_ptr = s.flat<float>().data();

        CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        QuantStats stats = QuantizationStats<V>(stream, SMs_, stats_ptr, x_ptr, sat_val_, ftz_val_, size);

        if (stats.max_val < max_stat_lo_) max_stat_lo_ = stats.max_val;
        if (stats.max_val > max_stat_hi_) max_stat_hi_ = stats.max_val;

        if (!logfile_.empty())
        {
          if (FILE* log = fopen(logfile_.c_str(), "a"))
          {
            float mean_stdv5 = stats.mean + stats.stdv * 5.0f;

            fprintf(log, "%.6f\t%.6f\t%3d\t%3d\t%3d\t%3d\t%3d\t%3d\t%d\t%s\n",
              stats.sat_pct,
              stats.ftz_pct,
              (*(int*)&stats.max_val >> 23)-127,
              (*(int*)&stats.mean    >> 23)-127,
              (*(int*)&stats.stdv    >> 23)-127,
              (*(int*)&mean_stdv5    >> 23)-127,
              (*(int*)&max_stat_lo_  >> 23)-127,
              (*(int*)&max_stat_hi_  >> 23)-127,
              step,
              this->name().c_str()
            );
            fclose(log);
          }
        }
      }
    }
  }
  int   SMs_, freq_, prev_step_;
  float sat_val_, ftz_val_, max_stat_hi_, max_stat_lo_;
  std::string logfile_;
  std::vector<int32> first_steps_;
};

REGISTER_KERNEL_BUILDER(Name("LogStats").Device(DEVICE_GPU).TypeConstraint<FLOAT>("T").HostMemory("step"),LogStatsOp<FLOAT,float>);
REGISTER_KERNEL_BUILDER(Name("LogStats").Device(DEVICE_GPU).TypeConstraint<BHALF>("T").HostMemory("step"),LogStatsOp<BHALF,bhalf>);
REGISTER_KERNEL_BUILDER(Name("LogStats").Device(DEVICE_GPU).TypeConstraint<EHALF>("T").HostMemory("step"),LogStatsOp<EHALF,ehalf>);
