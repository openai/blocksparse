
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/platform/stream_executor.h"
// #include "tensorflow/stream_executor/cuda/cuda_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <type_traits>
#include "gpu_types.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
// using perftools::gputools::cuda::AsCUDAStreamValue;

Status GetKernel(std::string& kernel_name, CUfunction* kernel);

Status BlocksparseConvShape(InferenceContext* ctx)
{
  int32 mode, C, K;
  std::vector<int32> DHW, MPQ, dimF;

  TF_RETURN_IF_ERROR(ctx->GetAttr("mode",   &mode));
  TF_RETURN_IF_ERROR(ctx->GetAttr("C",      &C));
  TF_RETURN_IF_ERROR(ctx->GetAttr("K",      &K));
  TF_RETURN_IF_ERROR(ctx->GetAttr("DHW",    &DHW));
  TF_RETURN_IF_ERROR(ctx->GetAttr("MPQ",    &MPQ));
  TF_RETURN_IF_ERROR(ctx->GetAttr("dimF",   &dimF));

  // fprop: NKMPQ = KCTRS . NCDHW
  if (mode == 0)
  {
    // a=F, b=I, c=O
    ShapeHandle I;
    DimensionHandle dh;
    TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(7), 3, &I));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 1), C, &dh));

    int32 rankI = ctx->Rank(I);
    if (rankI == 5)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[0], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), DHW[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 4), DHW[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(K), ctx->MakeDim(MPQ[0]), ctx->MakeDim(MPQ[1]), ctx->MakeDim(MPQ[2]) // NKMPQ
      }));
    }
    else if (rankI == 4)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), DHW[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(K), ctx->MakeDim(MPQ[1]), ctx->MakeDim(MPQ[2]) // NKPQ
      }));
    }
    else if (rankI == 3)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(K), ctx->MakeDim(MPQ[2])  // NKQ
      }));
    }
    else
      return errors::InvalidArgument("BlocksparseConv requires an input rank between 3 and 5: ", rankI);
  }
  // bprop: NCDHW = KCTRS . NKMPQ
  else if (mode == 1)
  {
    // a=F, b=I, c=O
    ShapeHandle I;
    DimensionHandle dh;
    TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(7), 3, &I));
    TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 1), K, &dh));

    int32 rankI = ctx->Rank(I);
    if (rankI == 5)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[0], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 4), MPQ[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(C), ctx->MakeDim(DHW[0]), ctx->MakeDim(DHW[1]), ctx->MakeDim(DHW[2]) // NCDHW
      }));
    }
    else if (rankI == 4)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), MPQ[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(C), ctx->MakeDim(DHW[1]), ctx->MakeDim(DHW[2]) // NCHW
      }));
    }
    else if (rankI == 3)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), MPQ[2], &dh));
      ctx->set_output(0, ctx->MakeShape({
        ctx->Dim(I, 0), ctx->MakeDim(C), ctx->MakeDim(DHW[2]) // NCW
      }));
    }
    else
      return errors::InvalidArgument("BlocksparseConv requires an input rank between 3 and 5: ", rankI);
  }
  // updat: KCTRS = NKMPQ . NCDHW
  else if (mode == 2)
  {
    // a=E, b=I, c=U
    ShapeHandle E, I;
    DimensionHandle dh;
    TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(6), 3, &E));
    int32 rankI = ctx->Rank(E);
    TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(7), rankI, &I));

    if (rankI == 5)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 2), MPQ[0], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 3), MPQ[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 4), MPQ[2], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[0], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), DHW[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 4), DHW[2], &dh));
    }
    else if (rankI == 4)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 2), MPQ[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 3), MPQ[2], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[1], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 3), DHW[2], &dh));
    }
    else if (rankI == 3)
    {
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(E, 2), MPQ[2], &dh));
      TF_RETURN_IF_ERROR(ctx->WithValue(ctx->Dim(I, 2), DHW[2], &dh));
    }
    else
      return errors::InvalidArgument("BlocksparseConv requires an input rank between 3 and 5: ", rankI);


    std::vector<DimensionHandle> dims;
    for (std::vector<int32>::iterator it = dimF.begin() ; it != dimF.end(); ++it)
      dims.emplace_back(ctx->MakeDim(*it));

    ctx->set_output(0, ctx->MakeShape( dims ));
  }
  return Status::OK();
}

REGISTER_OP("BlocksparseConv")
    .Input("fprop_grid: int32")
    .Input("bprop_grid: int32")
    .Input("updat_grid: int32")
    .Input("mpq_lut: int32")
    .Input("dhw_lut: int32")
    .Input("ck_lut: int32")
    .Input("a: a_type")
    .Input("b: b_type")
    .Output("c: c_type")
    .Attr("a_type: {half, float, bfloat16}")
    .Attr("b_type: {half, float, bfloat16}")
    .Attr("c_type: {half, float, bfloat16}")
    .Attr("mode: int >=0 =0")
    .Attr("overlapC: bool = false")
    .Attr("overlapK: bool = false")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("DHW: list(int) >= 3")
    .Attr("MPQ: list(int) >= 3")
    .Attr("dimF: list(int)")
    .Attr("trs: int >=0")
    .Attr("magic_trs: int >= 0")
    .Attr("shift_trs: int >= 0")
    .Attr("fshare: int >= 0")
    .Attr("bshare: int >= 0")
    .Attr("debug: bool = false")
    .SetShapeFn(BlocksparseConvShape)
    .Doc(R"doc(
Blocksparse convolution.
)doc");


template <typename AT, typename BT, typename CT>
class BlocksparseConvOp : public OpKernel {
 public:
  explicit BlocksparseConvOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode",      &mode_     ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("C",         &C_        ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("K",         &K_        ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("DHW",       &DHW_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("MPQ",       &MPQ_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimF",      &dimF_     ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trs",       &trs_      ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("magic_trs", &magic_trs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shift_trs", &shift_trs_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("debug",     &debug_    ));

    cdhw_ = C_ * DHW_[0] * DHW_[1] * DHW_[2];
    kmpq_ = K_ * MPQ_[0] * MPQ_[1] * MPQ_[2];
    zero_ = 0;
    size_f_ = 1;
    for (std::vector<int32>::iterator it = dimF_.begin() ; it != dimF_.end(); ++it)
      size_f_ *= *it;

    const char* dtypeA;
    if (mode_ == 2)
      dtypeA = std::is_same<AT, float>::value ? "E32" : "E16";
    else
      dtypeA = std::is_same<AT, float>::value ? "F32" : "F16";
    const char* dtypeB = std::is_same<BT, float>::value ? "I32" : "I16";
    const char* dtypeC = std::is_same<CT, float>::value ? "O32" : "O16";
    const char* overlap = "";
    const char* op;
    int depth;

    if (mode_  == 0)
    {
      bool overlapK;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(  "fshare",  &share_  ));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("overlapK",  &overlapK));
      op       = "fprop";
      depth    = 16;
      threads_ = 64;
      if (overlapK)
      {
        overlap = "_overlapK";
        zero_   = kmpq_ * sizeof(CT);
      }
    }
    else if (mode_  == 1)
    {
      bool overlapC;
      OP_REQUIRES_OK(ctx, ctx->GetAttr(  "bshare", &share_  ));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("overlapC", &overlapC));
      op       = "bprop";
      depth    = 16;
      threads_ = 64;
      if (overlapC)
      {
        overlap = "_overlapC";
        zero_   = cdhw_ * sizeof(CT);
      }
    }
    else //if (mode_  == 2)
    {
      op       = "updat";
      depth    = 32;
      zero_    = size_f_ * sizeof(CT);
      threads_ = 128;
      share_   = 0;
      overlap  = "";
    }
    char kernel_name[64];
    sprintf(kernel_name, "conv_blocksparse_32x32x%d_%s_%s_%s_%s%s", depth, op, dtypeA, dtypeB, dtypeC, overlap);
    kernel_name_ = kernel_name;
    kernel_ = 0;
  }

  virtual void Compute(OpKernelContext* ctx) override {

    const Tensor& grid_lut = ctx->input(mode_);
    const Tensor& mpq_lut  = ctx->input(mode_ == 1 ? 4 : 3);
    const Tensor& ck_lut   = ctx->input(5);
    const Tensor& a        = ctx->input(6);
    const Tensor& b        = ctx->input(7);

    float alpha = 1.0f;
    int   gridX = grid_lut.dim_size(0);
    int   rank  = b.dims();
    int   N     = b.dim_size(0);

    int zero, gridY;
    TensorShape c_shape;
    if (mode_ == 0)
    {
      zero  = N * zero_;
      gridY = N;
      c_shape.AddDim(N);
      c_shape.AddDim(K_);
      if (rank == 5) c_shape.AddDim(MPQ_[0]);
      if (rank >= 4) c_shape.AddDim(MPQ_[1]);
                     c_shape.AddDim(MPQ_[2]);
    }
    else if (mode_ == 1)
    {
      zero  = N * zero_;
      gridY = N;
      c_shape.AddDim(N);
      c_shape.AddDim(C_);
      if (rank == 5) c_shape.AddDim(DHW_[0]);
      if (rank >= 4) c_shape.AddDim(DHW_[1]);
                     c_shape.AddDim(DHW_[2]);
    }
    else
    {
      zero  = zero_;
      gridY = 1;
      for (std::vector<int32>::iterator it = dimF_.begin() ; it != dimF_.end(); ++it)
        c_shape.AddDim(*it);
    }

    Tensor* c = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &c));

    CUdeviceptr    c_ptr = (CUdeviceptr)c->flat<CT>().data();
    CUdeviceptr grid_ptr = (CUdeviceptr)grid_lut.flat<int32>().data();
    CUdeviceptr  mpq_ptr = (CUdeviceptr)mpq_lut.flat<int32>().data();
    CUdeviceptr   ck_ptr = (CUdeviceptr)ck_lut.flat<int32>().data();
    CUdeviceptr    a_ptr = (CUdeviceptr)a.flat<AT>().data();
    CUdeviceptr    b_ptr = (CUdeviceptr)b.flat<BT>().data();

    OP_REQUIRES_OK(ctx, GetKernel(kernel_name_, &kernel_));

    CUstream cu_stream = NULL; // AsCUDAStreamValue(ctx->op_device_context()->stream());

    void *args[] = {
      &grid_ptr, &mpq_ptr, &ck_ptr, &c_ptr, &a_ptr, &b_ptr, &alpha,
      &trs_, &magic_trs_, &shift_trs_, &cdhw_, &kmpq_, &N, &size_f_
    };

    CUresult res;
    if (zero > 0)
    {
      res = cuMemsetD8Async(c_ptr, 0, zero, cu_stream);
      if (res != CUDA_SUCCESS)
      {
        const char* errstr;
        cuGetErrorString(res, &errstr);
        OP_REQUIRES(ctx, false, errors::Internal("cuMemsetD8Async Error: ", errstr, " bytes: ", zero));
      }
    }
    res = cuLaunchKernel(kernel_, gridX, gridY, 1, threads_, 1, 1, share_, cu_stream, args, NULL);
    if (res != CUDA_SUCCESS)
    {
      const char* errstr;
      cuGetErrorString(res, &errstr);
      char params[256];
      sprintf(params, "m:%d(%5d,%5d:%5d), grid:%p, mpq:%p, ck:%p, c:%p, a:%p, b:%p, C:%5d, K:%5d, N:%3d %s\n",
        mode_, gridX, gridY, share_,
        (void*)grid_ptr, (void*)mpq_ptr, (void*)ck_ptr,
        (void*)c_ptr, (void*)a_ptr, (void*)b_ptr,
        C_, K_, N, kernel_name_.c_str());
      OP_REQUIRES(ctx, false, errors::Internal("cuLaunchKernel Error: ", errstr, "\nParams: ", params ));
      //OP_REQUIRES(ctx, false, errors::Internal("cuLaunchKernel Error: ", errstr));
    }
    if (debug_)
    {
      res = cuStreamSynchronize(cu_stream);
      if (res != CUDA_SUCCESS)
      {
        const char* errstr;
        cuGetErrorString(res, &errstr);

        char params[256];
        sprintf(params, "m:%d(%5d,%5d:%5d), grid:%p, mpq:%p, ck:%p, c:%p, a:%p, b:%p, C:%5d, K:%5d, N:%3d %s\n",
          mode_, gridX, gridY, share_,
          (void*)grid_ptr, (void*)mpq_ptr, (void*)ck_ptr,
          (void*)c_ptr, (void*)a_ptr, (void*)b_ptr,
          C_, K_, N, kernel_name_.c_str());

          OP_REQUIRES(ctx, false, errors::Internal("Cuda Error: ", errstr, "\nParams: ", params ));
      }
    }
  }
 private:
  int mode_, threads_, share_, zero_, C_, K_, trs_, magic_trs_, shift_trs_, size_f_, cdhw_, kmpq_;
  std::string kernel_name_;
  std::vector<int32> DHW_, MPQ_, dimF_;
  CUfunction kernel_;
  bool debug_;
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseConv").Device(DEVICE_GPU).TypeConstraint<float>("a_type").TypeConstraint<float>("b_type").TypeConstraint<float>("c_type"),BlocksparseConvOp<float, float, float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseConv").Device(DEVICE_GPU).TypeConstraint<EHALF>("a_type").TypeConstraint<EHALF>("b_type").TypeConstraint<EHALF>("c_type"),BlocksparseConvOp<EHALF, EHALF, EHALF>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseConv").Device(DEVICE_GPU).TypeConstraint<EHALF>("a_type").TypeConstraint<float>("b_type").TypeConstraint<float>("c_type"),BlocksparseConvOp<EHALF, float, float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseConv").Device(DEVICE_GPU).TypeConstraint<float>("a_type").TypeConstraint<EHALF>("b_type").TypeConstraint<EHALF>("c_type"),BlocksparseConvOp<float, EHALF, EHALF>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseConv").Device(DEVICE_GPU).TypeConstraint<float>("a_type").TypeConstraint<EHALF>("b_type").TypeConstraint<float>("c_type"),BlocksparseConvOp<float, EHALF, float>);

REGISTER_OP("BlocksparseDeconv")
    .Input("fprop_grid: int32")
    .Input("bprop_grid: int32")
    .Input("updat_grid: int32")
    .Input("mpq_lut: int32")
    .Input("dhw_lut: int32")
    .Input("ck_lut: int32")
    .Input("a: a_type")
    .Input("b: b_type")
    .Output("c: c_type")
    .Attr("a_type: {half, float, bfloat16}")
    .Attr("b_type: {half, float, bfloat16}")
    .Attr("c_type: {half, float, bfloat16}")
    .Attr("mode: int >=0 =0")
    .Attr("overlapC: bool = false")
    .Attr("overlapK: bool = false")
    .Attr("C: int >=0")
    .Attr("K: int >=0")
    .Attr("DHW: list(int) >= 3")
    .Attr("MPQ: list(int) >= 3")
    .Attr("dimF: list(int)")
    .Attr("trs: int >=0")
    .Attr("magic_trs: int >= 0")
    .Attr("shift_trs: int >= 0")
    .Attr("fshare: int >= 0")
    .Attr("bshare: int >= 0")
    .Attr("debug: bool = false")
    .SetShapeFn(BlocksparseConvShape)
    .Doc(R"doc(
Blocksparse convolution.
)doc");

template <typename AT, typename BT, typename CT>
class BlocksparseDeconvOp : public BlocksparseConvOp<AT,BT,CT> {
 public:
  explicit BlocksparseDeconvOp(OpKernelConstruction* ctx) : BlocksparseConvOp<AT,BT,CT>(ctx) {}
};

REGISTER_KERNEL_BUILDER(Name("BlocksparseDeconv").Device(DEVICE_GPU).TypeConstraint<float>("a_type").TypeConstraint<float>("b_type").TypeConstraint<float>("c_type"),BlocksparseDeconvOp<float, float, float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseDeconv").Device(DEVICE_GPU).TypeConstraint<EHALF>("a_type").TypeConstraint<EHALF>("b_type").TypeConstraint<EHALF>("c_type"),BlocksparseDeconvOp<EHALF, EHALF, EHALF>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseDeconv").Device(DEVICE_GPU).TypeConstraint<EHALF>("a_type").TypeConstraint<float>("b_type").TypeConstraint<float>("c_type"),BlocksparseDeconvOp<EHALF, float, float>);
REGISTER_KERNEL_BUILDER(Name("BlocksparseDeconv").Device(DEVICE_GPU).TypeConstraint<EHALF>("a_type").TypeConstraint<float>("b_type").TypeConstraint<EHALF>("c_type"),BlocksparseDeconvOp<EHALF, float, EHALF>);
