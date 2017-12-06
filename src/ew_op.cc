
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

template <typename T, typename V> bool EW_Forward(CUstream stream, T* z, const T* x, const T* y, const float* b, float alpha, int size, int N, int op);
template <typename B, typename F, typename VB, typename VF> bool EW_Backward(CUstream stream, B* dx, B* dy, float* db, const B* dz, const F* x, const F* y, const F* z, const float* g, float alpha, int size, int N, int op);
template <typename T, typename V> bool LSTM_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c_prev, const T* h_prev, int N, int K);
template <typename T, typename V> bool LSTM4_Gates_Forward(CUstream stream, T* c_next, T* h_next, const T* c, const T* i, const T* f, const T* o, const T* u, int N, int K);
template <typename B, typename F, typename VB, typename VF> bool LSTM_Gates_Backward(CUstream stream, B* dc, B* dh, const B* ec, const B* eh, const F* c_prev, const F* h_prev, int N, int K);
template <typename B, typename A, typename VB, typename VA> bool LSTM4_Gates_Backward(CUstream stream, B* dc, B* di, B* df, B* doo, B* du, const B* ec, const B* eh, const A* c, const A* i, const A* f, const A* o, const A* u, int N, int K);
template <typename T, typename V> bool Split4_Forward (CUstream stream, T* z0, T* z1, T* z2, T* z3, const T* x, int N, int K);
template <typename T, typename V> bool Concat4_Forward(CUstream stream, T* dx, const T* z0, const T* z1, const T* z2, const T* z3, int N, int K);
template <typename TY, typename TX, typename VY, typename VX> bool FloatCast(CUstream stream, TY* y, const TX* x, int size);
template <typename T, typename V> bool SparseReluForward(CUstream stream, T* y, const T* x, float alpha, int K, int N);
template <typename T, typename V> bool DropoutForward(CUstream stream, T* y, char* m, const T* x,int size, float keep_prob);
template <typename T, typename V> bool DropoutBackward(CUstream stream,T* dx, const char* m, const T* dy,int size);
template <typename T, typename V> bool DropoutMaskForward(CUstream stream, T* y, const char* m, const T* x,int size);
template <typename T, typename V> bool AddN(CUstream stream, struct plist8<T>* x, T* z, int size, int params);

Status UnchangedShape(shape_inference::InferenceContext* ctx);


REGISTER_OP("EwZXy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
binary elementwise op
)doc");

template <typename T, typename V1, typename V4>
class EwZXyOp : public OpKernel {
 public:
  explicit EwZXyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& y = ctx->input(1);

    int size = x.shape().num_elements();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

          V1* z_ptr = (V1*)z->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();
    const V1* y_ptr = (const V1*)y.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, y_ptr, 0, 1.0f, size, 0, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXyOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXyOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXy").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXyOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwZXa")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise op (with optional alpha)
)doc");

template <typename T, typename V1, typename V4>
class EwZXaOp : public OpKernel {
 public:
  explicit EwZXaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int size = x.shape().num_elements();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

          V1* z_ptr = (V1*)z->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, 0, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXaOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXaOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXa").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXaOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwZXb")
    .Input("x: T")
    .Input("b: float")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
broadcast elementwise op
)doc");

template <typename T, typename V1, typename V4>
class EwZXbOp : public OpKernel {
 public:
  explicit EwZXbOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& b = ctx->input(1);

    int rank = x.dims();
    int K    = x.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

             V1* z_ptr = (V1*)z->flat<T>().data();
    const    V1* x_ptr = (const V1*)x.flat<T>().data();
    const float* b_ptr = b.flat<float>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Forward<V1,V4>(stream, z_ptr, x_ptr, 0, b_ptr, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<float>("T"),EwZXbOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),EwZXbOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwZXb").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),EwZXbOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("EwDxdyDzxy")
    .Input("dz: B")
    .Input("x: F")
    .Input("y: F")
    .Output("dx: B")
    .Output("dy: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
binary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxdyDzxyOp : public OpKernel {
 public:
  explicit EwDxdyDzxyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);
    const Tensor&  y = ctx->input(2);

    int size = x.shape().num_elements();

    Tensor* dx = nullptr;
    Tensor* dy = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, y.shape(), &dy));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
          VB1* dy_ptr = (VB1*)dy->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  x_ptr = (const VF1*)x.flat<F>().data();
    const VF1*  y_ptr = (const VF1*)y.flat<F>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, dy_ptr, 0, dz_ptr, x_ptr, y_ptr, 0, 0, 1.0f, size, 0, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxdyDzxyOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxdyDzxyOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxdyDzxyOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxdyDzxyOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdyDzxy").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxdyDzxyOp<float,BHALF,float,bhalf,float4,bhalf4>);

REGISTER_OP("EwDxDzza")
    .Input("dz: B")
    .Input("z: F")
    .Output("dx: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxDzzaOp : public OpKernel {
 public:
  explicit EwDxDzzaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  z = ctx->input(1);

    int size = z.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, z.shape(), &dx));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  z_ptr = (const VF1*)z.flat<F>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, 0, dz_ptr, 0, 0, z_ptr, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxDzzaOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxDzzaOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxDzzaOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxDzzaOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzza").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxDzzaOp<float,BHALF,float,bhalf,float4,bhalf4>);


REGISTER_OP("EwDxDzxa")
    .Input("dz: B")
    .Input("x: F")
    .Output("dx: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .Attr("op: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxDzxaOp : public OpKernel {
 public:
  explicit EwDxDzxaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);

    int size = x.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));

          VB1* dx_ptr = (VB1*)dx->flat<B>().data();
    const VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const VF1*  x_ptr = (const VF1*)x.flat<F>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, 0, dz_ptr, x_ptr, 0, 0, 0, alpha_, size, 0, op_);
  }
  int op_;
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxDzxaOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxDzxaOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxDzxaOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxDzxaOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxDzxa").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxDzxaOp<float,BHALF,float,bhalf,float4,bhalf4>);


REGISTER_OP("EwDbDzb")
    .Input("dz: B")
    .Input("b: float")
    .Output("db: float")
    .Attr("B: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename VB1, typename VB4>
class EwDbDzbOp : public OpKernel {
 public:
  explicit EwDbDzbOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  b = ctx->input(1);

    int rank = dz.dims();
    int K    = dz.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= dz.dim_size(--rank);

    Tensor* db = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, b.shape(), &db));

          float* db_ptr = db->flat<float>().data();
    const   VB1* dz_ptr = (const VB1*)dz.flat<B>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Backward<VB1,VB1,VB4,VB4>(stream, 0, 0, db_ptr, dz_ptr, 0, 0, 0, 0, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<float>("B"),EwDbDzbOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<EHALF>("B"),EwDbDzbOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDbDzb").Device(DEVICE_GPU).TypeConstraint<BHALF>("B"),EwDbDzbOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("EwDxdgDzxg")
    .Input("dz: B")
    .Input("x: F")
    .Input("g: float")
    .Output("dx: B")
    .Output("dg: float")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("op: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(1));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
unary elementwise grad op
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class EwDxdgDzxgOp : public OpKernel {
 public:
  explicit EwDxdgDzxgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz = ctx->input(0);
    const Tensor&  x = ctx->input(1);
    const Tensor&  g = ctx->input(2);

    int rank = x.dims();
    int K    = x.dim_size(--rank);
    int N    = 1;
    while (rank > 0) N *= x.dim_size(--rank);

    Tensor* dx = nullptr;
    Tensor* dg = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &dx));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, g.shape(), &dg));

            VB1* dx_ptr = (VB1*)dx->flat<B>().data();
          float* dg_ptr = dg->flat<float>().data();
    const   VB1* dz_ptr = (const VB1*)dz.flat<B>().data();
    const   VF1*  x_ptr = (const VF1*)x.flat<F>().data();
    const float*  g_ptr = g.flat<float>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    EW_Backward<VB1,VF1,VB4,VF4>(stream, dx_ptr, 0, dg_ptr, dz_ptr, x_ptr, 0, 0, g_ptr, 1.0f, K, N, op_);
  }
  int op_;
};
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),EwDxdgDzxgOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),EwDxdgDzxgOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),EwDxdgDzxgOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),EwDxdgDzxgOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("EwDxdgDzxg").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),EwDxdgDzxgOp<float,BHALF,float,bhalf,float4,bhalf4>);



REGISTER_OP("LSTMGates")
    .Input("c_prev: T")
    .Input("h_prev: T")
    .Output("c_next: T")
    .Output("h_next: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
  i, f, o, u = split(z, 4, 1)
  i = sigmoid(i)
  f = sigmoid(f)
  o = sigmoid(o)
  u = tanh(u)
  c = add(multiply(f, c), multiply(i, u))
  h = multiply(o, tanh(c))
)doc");

template <typename T, typename V1, typename V4>
class LSTMGatesOp : public OpKernel {
 public:
  explicit LSTMGatesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& c_prev = ctx->input(0);
    const Tensor& h_prev = ctx->input(1);

    int N = h_prev.dim_size(0);
    int K = h_prev.dim_size(1);

    Tensor* c_next = nullptr;
    Tensor* h_next = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_prev.shape(), &c_next));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, c_prev.shape(), &h_next));

          V1* c_next_ptr = (      V1*)c_next->flat<T>().data();
          V1* h_next_ptr = (      V1*)h_next->flat<T>().data();
    const V1* c_prev_ptr = (const V1*)c_prev.flat<T>().data();
    const V1* h_prev_ptr = (const V1*)h_prev.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    LSTM_Gates_Forward<V1,V4>(stream, c_next_ptr, h_next_ptr, c_prev_ptr, h_prev_ptr, N,  K);
  }
};
REGISTER_KERNEL_BUILDER(Name("LSTMGates").Device(DEVICE_GPU).TypeConstraint<float>("T"),LSTMGatesOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),LSTMGatesOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),LSTMGatesOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("LSTMGates4")
    .Input("c: T")
    .Input("i: T")
    .Input("f: T")
    .Input("o: T")
    .Input("u: T")
    .Output("c_next: T")
    .Output("h_next: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
  i = sigmoid(i)
  f = sigmoid(f)
  o = sigmoid(o)
  u = tanh(u)
  c = add(multiply(f, c), multiply(i, u))
  h = multiply(o, tanh(c))
)doc");

template <typename T, typename V1, typename V4 >
class LSTMGates4Op : public OpKernel {
 public:
  explicit LSTMGates4Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& c = ctx->input(0);
    const Tensor& i = ctx->input(1);
    const Tensor& f = ctx->input(2);
    const Tensor& o = ctx->input(3);
    const Tensor& u = ctx->input(4);

    int N = c.dim_size(0);
    int K = c.dim_size(1);

    Tensor* c_next = nullptr;
    Tensor* h_next = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c.shape(), &c_next));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, c.shape(), &h_next));

    V1* c_next_ptr = (V1*)c_next->flat<T>().data();
    V1* h_next_ptr = (V1*)h_next->flat<T>().data();
    const V1* c_ptr = (const V1*)c.flat<T>().data();
    const V1* i_ptr = (const V1*)i.flat<T>().data();
    const V1* f_ptr = (const V1*)f.flat<T>().data();
    const V1* o_ptr = (const V1*)o.flat<T>().data();
    const V1* u_ptr = (const V1*)u.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    LSTM4_Gates_Forward<V1,V4>(stream, c_next_ptr, h_next_ptr, c_ptr, i_ptr, f_ptr, o_ptr, u_ptr, N,  K);
  }
};
REGISTER_KERNEL_BUILDER(Name("LSTMGates4").Device(DEVICE_GPU).TypeConstraint<float>("T"),LSTMGates4Op<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),LSTMGates4Op<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),LSTMGates4Op<BHALF,bhalf,bhalf4>);

REGISTER_OP("LSTMGatesGrad")
    .Input("c_prev: F")
    .Input("h_prev: F")
    .Input("grads: N * B")
    .Output("dc: B")
    .Output("dh: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("N: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
  di = ec * tan_u * sig_grad(sig_i)
  df = ec * c * sig_grad(sig_f)
  do = eh * c_act * sig_grad(sig_o)
  du = ec * sig_i * tanh_grad(tan_u)
  dc = (ec  +  eh * sig_o * tanh_grad(c_act)) * sig_f
  dh = concat([di,df,do,du], 1)
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class LSTMGatesGradOp : public OpKernel {
 public:
  explicit LSTMGatesGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& c_prev = ctx->input(0);
    const Tensor& h_prev = ctx->input(1);
    const Tensor&     eh = ctx->input(2);

    int N = h_prev.dim_size(0);
    int K = h_prev.dim_size(1);

    Tensor* dc = nullptr;
    Tensor* dh = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_prev.shape(), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, h_prev.shape(), &dh));

    VB1* dc_ptr = (VB1*)dc->flat<B>().data();
    VB1* dh_ptr = (VB1*)dh->flat<B>().data();
    const VF1* c_prev_ptr = (const VF1*)c_prev.flat<F>().data();
    const VF1* h_prev_ptr = (const VF1*)h_prev.flat<F>().data();
    const VB1*     eh_ptr = (const VB1*)eh.flat<B>().data();
    const VB1*     ec_ptr = nullptr;
    if (ctx->num_inputs() == 4)
      ec_ptr = (const VB1*)ctx->input(3).flat<B>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    LSTM_Gates_Backward<VB1,VF1,VB4,VF4>(stream, dc_ptr, dh_ptr, ec_ptr, eh_ptr, c_prev_ptr, h_prev_ptr, N, K);
  }
};
REGISTER_KERNEL_BUILDER(Name("LSTMGatesGrad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),LSTMGatesGradOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGatesGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),LSTMGatesGradOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGatesGrad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),LSTMGatesGradOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGatesGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),LSTMGatesGradOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGatesGrad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),LSTMGatesGradOp<float,BHALF,float,bhalf,float4,bhalf4>);


REGISTER_OP("LSTMGates4Grad")
    .Input("c: F")
    .Input("i: F")
    .Input("f: F")
    .Input("o: F")
    .Input("u: F")
    .Input("grads: N * B")
    .Output("dc: B")
    .Output("di: B")
    .Output("df: B")
    .Output("do: B")
    .Output("du: B")
    .Attr("B: {half, float, bfloat16}")
    .Attr("F: {half, float, bfloat16}")
    .Attr("N: int")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      ctx->set_output(2, ctx->input(0));
      ctx->set_output(3, ctx->input(0));
      ctx->set_output(4, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
  di = ec * tan_u * sig_grad(sig_i)
  df = ec * c * sig_grad(sig_f)
  do = eh * c_act * sig_grad(sig_o)
  du = ec * sig_i * tanh_grad(tan_u)
  dc = (ec  +  eh * sig_o * tanh_grad(c_act)) * sig_f
)doc");

template <typename B, typename F, typename VB1, typename VF1, typename VB4, typename VF4>
class LSTMGates4GradOp : public OpKernel {
 public:
  explicit LSTMGates4GradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& c  = ctx->input(0);
    const Tensor& i  = ctx->input(1);
    const Tensor& f  = ctx->input(2);
    const Tensor& o  = ctx->input(3);
    const Tensor& u  = ctx->input(4);
    const Tensor& eh = ctx->input(5);

    int N = c.dim_size(0);
    int K = c.dim_size(1);

    Tensor* dc  = nullptr;
    Tensor* di  = nullptr;
    Tensor* df  = nullptr;
    Tensor* doo = nullptr;
    Tensor* du  = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c.shape(), &dc));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, c.shape(), &di));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, c.shape(), &df));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, c.shape(), &doo));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, c.shape(), &du));

    VB1* dc_ptr = (VB1*)dc->flat<B>().data();
    VB1* di_ptr = (VB1*)di->flat<B>().data();
    VB1* df_ptr = (VB1*)df->flat<B>().data();
    VB1* do_ptr = (VB1*)doo->flat<B>().data();
    VB1* du_ptr = (VB1*)du->flat<B>().data();
    const VF1*  c_ptr = (const VF1*)c.flat<F>().data();
    const VF1*  i_ptr = (const VF1*)i.flat<F>().data();
    const VF1*  f_ptr = (const VF1*)f.flat<F>().data();
    const VF1*  o_ptr = (const VF1*)o.flat<F>().data();
    const VF1*  u_ptr = (const VF1*)u.flat<F>().data();
    const VB1* eh_ptr = (const VB1*)eh.flat<B>().data();
    const VB1* ec_ptr = nullptr;
    if (ctx->num_inputs() == 7)
      ec_ptr = (const VB1*)ctx->input(6).flat<B>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    LSTM4_Gates_Backward<VB1,VF1,VB4,VF4>(stream, dc_ptr, di_ptr, df_ptr, do_ptr, du_ptr, ec_ptr, eh_ptr, c_ptr, i_ptr, f_ptr, o_ptr, u_ptr, N, K);
  }
};
REGISTER_KERNEL_BUILDER(Name("LSTMGates4Grad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<float>("F"),LSTMGates4GradOp<float,float,float,float,float4,float4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4Grad").Device(DEVICE_GPU).TypeConstraint<EHALF>("B").TypeConstraint<EHALF>("F"),LSTMGates4GradOp<EHALF,EHALF,ehalf,ehalf,ehalf4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4Grad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<EHALF>("F"),LSTMGates4GradOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4Grad").Device(DEVICE_GPU).TypeConstraint<BHALF>("B").TypeConstraint<BHALF>("F"),LSTMGates4GradOp<BHALF,BHALF,bhalf,bhalf,bhalf4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("LSTMGates4Grad").Device(DEVICE_GPU).TypeConstraint<float>("B").TypeConstraint<BHALF>("F"),LSTMGates4GradOp<float,BHALF,float,bhalf,float4,bhalf4>);



REGISTER_OP("Split4")
    .Input("x: T")
    .Output("z0: T")
    .Output("z1: T")
    .Output("z2: T")
    .Output("z3: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle x = ctx->input(0);
      DimensionHandle N = ctx->Dim(x,0);
      DimensionHandle K;
      TF_RETURN_IF_ERROR(ctx->Divide( ctx->Dim(x,1), 4, true, &K));
      ShapeHandle z = ctx->MakeShape({ N, K });
      ctx->set_output(0, z);
      ctx->set_output(1, z);
      ctx->set_output(2, z);
      ctx->set_output(3, z);
      return Status::OK();
    })
    .Doc(R"doc(
split 4 for lstm type nets
)doc");

template <typename T, typename V1, typename V4>
class Split4Op : public OpKernel {
 public:
  explicit Split4Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int N = x.dim_size(0);
    int K = x.dim_size(1);

    TensorShape z_shape({N, K >> 2});

    Tensor* z0 = nullptr;
    Tensor* z1 = nullptr;
    Tensor* z2 = nullptr;
    Tensor* z3 = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, z_shape, &z0));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, z_shape, &z1));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, z_shape, &z2));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, z_shape, &z3));

    V1* z0_ptr = (V1*)z0->flat<T>().data();
    V1* z1_ptr = (V1*)z1->flat<T>().data();
    V1* z2_ptr = (V1*)z2->flat<T>().data();
    V1* z3_ptr = (V1*)z3->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    Split4_Forward<V1,V4>(stream, z0_ptr, z1_ptr, z2_ptr, z3_ptr, x_ptr, N,  K);
  }
};
REGISTER_KERNEL_BUILDER(Name("Split4").Device(DEVICE_GPU).TypeConstraint<float>("T"),Split4Op<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Split4").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),Split4Op<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Split4").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),Split4Op<BHALF,bhalf,bhalf4>);


REGISTER_OP("Concat4")
    .Input("dz0: T")
    .Input("dz1: T")
    .Input("dz2: T")
    .Input("dz3: T")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle dz = ctx->input(0);
      DimensionHandle N = ctx->Dim(dz,0);
      DimensionHandle K;
      TF_RETURN_IF_ERROR(ctx->Multiply( ctx->Dim(dz,1), 4, &K));
      ShapeHandle dx = ctx->MakeShape({ N, K });
      ctx->set_output(0, dx);
      return Status::OK();
    })
    .Doc(R"doc(
split 4 grad for lstm type nets
)doc");

template <typename T, typename V1, typename V4>
class Concat4Op : public OpKernel {
 public:
  explicit Concat4Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dz0 = ctx->input(0);
    const Tensor& dz1 = ctx->input(1);
    const Tensor& dz2 = ctx->input(2);
    const Tensor& dz3 = ctx->input(3);

    int N = dz0.dim_size(0);
    int K = dz0.dim_size(1) << 2;

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({N, K}), &dx));

    V1* dx_ptr = (V1*)dx->flat<T>().data();
    const V1* dz0_ptr = (const V1*)dz0.flat<T>().data();
    const V1* dz1_ptr = (const V1*)dz1.flat<T>().data();
    const V1* dz2_ptr = (const V1*)dz2.flat<T>().data();
    const V1* dz3_ptr = (const V1*)dz3.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    Concat4_Forward<V1,V4>(stream, dx_ptr, dz0_ptr, dz1_ptr, dz2_ptr, dz3_ptr, N, K);
  }
};
REGISTER_KERNEL_BUILDER(Name("Concat4").Device(DEVICE_GPU).TypeConstraint<float>("T"),Concat4Op<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Concat4").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),Concat4Op<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Concat4").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),Concat4Op<BHALF,bhalf,bhalf4>);



REGISTER_OP("FloatCast")
    .Input("x: TX")
    .Output("y: TY")
    .Attr("TX: {half, float, bfloat16}")
    .Attr("TY: {half, float, bfloat16}")
    .Attr("dx_dtype: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
float to half and half to float
)doc");

template <typename TY, typename TX, typename VY1, typename VX1, typename VY4, typename VX4>
class FloatCastOp : public OpKernel {
 public:
  explicit FloatCastOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int size = x.shape().num_elements();

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

          VY1* y_ptr = (VY1*)y->flat<TY>().data();
    const VX1* x_ptr = (const VX1*)x.flat<TX>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    FloatCast<VY1,VX1,VY4,VX4>(stream, y_ptr, x_ptr, size);
  }
};
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<float>("TY").TypeConstraint<EHALF>("TX"),FloatCastOp<float,EHALF,float,ehalf,float4,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<EHALF>("TY").TypeConstraint<float>("TX"),FloatCastOp<EHALF,float,ehalf,float,ehalf4,float4>);

REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<   float>("TY").TypeConstraint<bfloat16>("TX"),FloatCastOp<float,bfloat16,float,bhalf,float4,bhalf4>);
REGISTER_KERNEL_BUILDER(Name("FloatCast").Device(DEVICE_GPU).TypeConstraint<bfloat16>("TY").TypeConstraint<   float>("TX"),FloatCastOp<bfloat16,float,bhalf,float,bhalf4,float4>);



REGISTER_OP("SparseRelu")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("alpha: float = 1.0")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
sparse relu computed from mean/std
)doc");

template <typename T, typename V1, typename V4>
class SparseReluOp : public OpKernel {
 public:
  explicit SparseReluOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int N = x.dim_size(0);
    int K = x.dim_size(1);

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &z));

          V1* z_ptr = (V1*)z->flat<T>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    SparseReluForward<V1,V4>(stream, z_ptr, x_ptr, alpha_, K, N);
  }
  float alpha_;
};
REGISTER_KERNEL_BUILDER(Name("SparseRelu").Device(DEVICE_GPU).TypeConstraint<float>("T"),SparseReluOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("SparseRelu").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),SparseReluOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("SparseRelu").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),SparseReluOp<BHALF,bhalf,bhalf4>);



REGISTER_OP("Dropout")
    .Input("x: T")
    .Output("y: T")
    .Output("m: int8")
    .Attr("T: {half, float, bfloat16}")
    .Attr("keep_prob: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Simple elementwise dropout.
Random source is clock + tid/bid + tousworthe generator.
)doc");

template <typename T, typename V1, typename V4>
class DropoutOp : public OpKernel {
 public:
  explicit DropoutOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_prob", &keep_prob_));

    OP_REQUIRES(ctx, keep_prob_ >= 0.0f && keep_prob_ <= 1.0f, errors::Internal("Dropout: 0 <= keep_prob <= 1"));

  }
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);

    int size = x.shape().num_elements();

    OP_REQUIRES(ctx, (size & 3) == 0, errors::Internal("Dropout tensor size must be multiple of 4"));

    Tensor* y = nullptr;
    Tensor* m = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, x.shape(), &m));

          V1* y_ptr = (V1*)y->flat<T>().data();
        char* m_ptr = (char*)m->flat<int8>().data();
    const V1* x_ptr = (const V1*)x.flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    DropoutForward<V1,V4>(stream, y_ptr, m_ptr, x_ptr, size, keep_prob_);
  }
  float keep_prob_;
};
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<float>("T"),DropoutOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),DropoutOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("Dropout").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),DropoutOp<BHALF,bhalf,bhalf4>);

REGISTER_OP("DropoutMask")
    .Input("x: T")
    .Input("m: int8")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Simple elementwise dropout with previously computed mask.
)doc");

template <typename T, typename V1, typename V4>
class DropoutMaskOp : public OpKernel {
 public:
  explicit DropoutMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x = ctx->input(0);
    const Tensor& m = ctx->input(1);

    int size = x.shape().num_elements();

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

            V1* y_ptr = (      V1*)y->flat<T>().data();
    const   V1* x_ptr = (const V1*)x.flat<T>().data();
    const char* m_ptr = (const char*)m.flat<int8>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    DropoutMaskForward<V1,V4>(stream, y_ptr, m_ptr, x_ptr, size);
  }
};
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<float>("T"),DropoutMaskOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),DropoutMaskOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("DropoutMask").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),DropoutMaskOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("DropoutGrad")
    .Input("dy: T")
    .Input("m: int8")
    .Output("dx: T")
    .Attr("T: {half, float, bfloat16}")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
Simple elementwise dropout gradient.
)doc");

template <typename T, typename V1, typename V4>
class DropoutGradOp : public OpKernel {
 public:
  explicit DropoutGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& dy = ctx->input(0);
    const Tensor& m  = ctx->input(1);

    int size = dy.shape().num_elements();

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dy.shape(), &dx));

            V1* dx_ptr = (      V1*)dx->flat<T>().data();
    const   V1* dy_ptr = (const V1*)dy.flat<T>().data();
    const char* m_ptr  = (const char*)m.flat<int8>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    DropoutBackward<V1,V4>(stream, dx_ptr, m_ptr, dy_ptr, size);
  }
};
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),DropoutGradOp<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),DropoutGradOp<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),DropoutGradOp<BHALF,bhalf,bhalf4>);


REGISTER_OP("AddN8")
    .Input("x8: N * T")
    .Output("y: T")
    .Attr("T: {half, float, bfloat16}")
    .Attr("N: int")
    .SetShapeFn(UnchangedShape)
    .Doc(R"doc(
  y = x[0] + x[1] + x[2] ... x[7]
)doc");

template <typename T, typename V1, typename V4>
class AddN8Op : public OpKernel {
 public:
  explicit AddN8Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {

    const Tensor& x0 = ctx->input(0);

    int params = ctx->num_inputs();
    int size   = x0.shape().num_elements();

    struct plist8<V1> x8;
    for (int i = 0; i < params; ++i)
        x8.a[i] = (const V1*)ctx->input(i).flat<T>().data();

    Tensor* z = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x0.shape(), &z));
    V1* z_ptr = (V1*)z->flat<T>().data();

    CUstream stream = NULL; //  AsCUDAStreamValue(ctx->op_device_context()->stream());

    AddN<V1,V4>(stream, &x8, z_ptr, size, params);
  }
};
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<float>("T"),AddN8Op<float,float,float4>);
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<EHALF>("T"),AddN8Op<EHALF,ehalf,ehalf4>);
REGISTER_KERNEL_BUILDER(Name("AddN8").Device(DEVICE_GPU).TypeConstraint<BHALF>("T"),AddN8Op<BHALF,bhalf,bhalf4>);


