
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from blocksparse.utils import _op_module, get_entropy, scalar_constant


ew_z_xy_op      = _op_module.ew_z_xy
ew_z_xa_op      = _op_module.ew_z_xa

ew_dxdy_dzxy_op = _op_module.ew_dxdy_dzxy
ew_dx_dzxa_op   = _op_module.ew_dx_dzxa
ew_dx_dzza_op   = _op_module.ew_dx_dzza

ew_z_xb_op      = _op_module.ew_z_xb
ew_db_dzb_op    = _op_module.ew_db_dzb
ew_dxdg_dzxg_op = _op_module.ew_dxdg_dzxg

ADD_OP     =  0
SUB_OP     =  1
MUL_OP     =  2
DIV_OP     =  3
MAXIMUM_OP =  4
MINIMUM_OP =  5
NEG_OP     =  6
RCP_OP     =  7
SQR_OP     =  8
SQRT_OP    =  9
EXP_OP     = 10
LOG_OP     = 11
SIG_OP     = 12
TANH_OP    = 13
RELU_OP    = 14
ELU_OP     = 15
GELU_OP    = 16
SWISH_OP   = 17
BIASADD_OP = 18
GAINMUL_OP = 19

ew_names = [
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Maximum",
    "Minimum",
    "Neg",
    "Rcp",
    "Sqr",
    "Sqrt",
    "Exp",
    "Log",
    "Sig",
    "Tanh",
    "Relu",
    "Elu",
    "Gelu",
    "Swish",
    "Biasadd",
    "Gainmul",
]

def broadcast_check(x, y, ew_op, bc_op, tf_op, name):
    xshape = x.shape.as_list()
    yshape = y.shape.as_list()

    if xshape == yshape:
        if name is None: name = ew_names[ew_op]
        return ew_z_xy_op(x, y, op=ew_op, name=name)

    if bc_op is not None:
        if xshape[-1] == yshape[-1]:
            if yshape[-1] == y.get_shape().num_elements():
                if name is None: name = ew_names[bc_op]
                return ew_z_xb_op(x, y, op=bc_op, name=name)

            if xshape[-1] == x.get_shape().num_elements():
                if name is None: name = ew_names[bc_op]
                return ew_z_xb_op(y, x, op=bc_op, name=name)

    # fall back to tf for everything else for now..
    return tf_op(x, y, name=name)

def        add(x, y, name=None): return broadcast_check(x, y,     ADD_OP, BIASADD_OP, tf.add,      name)
def   multiply(x, y, name=None): return broadcast_check(x, y,     MUL_OP, GAINMUL_OP, tf.multiply, name)
def   subtract(x, y, name=None): return broadcast_check(x, y,     SUB_OP,       None, tf.subtract, name)
def     divide(x, y, name=None): return broadcast_check(x, y,     DIV_OP,       None, tf.divide,   name)
def    maximum(x, y, name=None): return broadcast_check(x, y, MAXIMUM_OP,       None, tf.maximum,  name)
def    minimum(x, y, name=None): return broadcast_check(x, y, MINIMUM_OP,       None, tf.minimum,  name)

def   negative(x,    name=None): return ew_z_xa_op(x, op= NEG_OP, name=ew_names[ NEG_OP] if name is None else name)
def reciprocal(x,    name=None): return ew_z_xa_op(x, op= RCP_OP, name=ew_names[ RCP_OP] if name is None else name)
def     square(x,    name=None): return ew_z_xa_op(x, op= SQR_OP, name=ew_names[ SQR_OP] if name is None else name)
def       sqrt(x,    name=None): return ew_z_xa_op(x, op=SQRT_OP, name=ew_names[SQRT_OP] if name is None else name)
def        exp(x,    name=None): return ew_z_xa_op(x, op= EXP_OP, name=ew_names[ EXP_OP] if name is None else name)
def        log(x,    name=None): return ew_z_xa_op(x, op= LOG_OP, name=ew_names[ LOG_OP] if name is None else name)
def    sigmoid(x,    name=None): return ew_z_xa_op(x, op= SIG_OP, name=ew_names[ SIG_OP] if name is None else name)
def       tanh(x,    name=None): return ew_z_xa_op(x, op=TANH_OP, name=ew_names[TANH_OP] if name is None else name)
def       relu(x,    name=None): return ew_z_xa_op(x, op=RELU_OP, name=ew_names[RELU_OP] if name is None else name)

# WARNING: gelu op, not numerically stable... need to investigate more.  Use fast_gelu for now.

def elu   (x, alpha=1.0,      name=None): return ew_z_xa_op(x, op=ELU_OP,   alpha=alpha, name=ew_names[ELU_OP]   if name is None else name)
def gelu  (x, alpha=0.044715, name=None): return ew_z_xa_op(x, op=GELU_OP,  alpha=alpha, name=ew_names[GELU_OP]  if name is None else name)
def swish (x, alpha=1.0,      name=None): return ew_z_xa_op(x, op=SWISH_OP, alpha=alpha, name=ew_names[SWISH_OP] if name is None else name)

def fast_gelu(x, name=None):
    return swish(x, alpha=1.702, name=name)

@ops.RegisterGradient("EwZXy")
def ew_z_xy_grad(op, dz):
    op_code = op.get_attr("op")
    name    = ew_names[op_code] + "_grad"

    if op_code == ADD_OP:
        return (dz, dz)

    if op_code == SUB_OP:
        return (dz, ew_z_xa_op(dz, op=NEG_OP, name=name))

    return ew_dxdy_dzxy_op(dz, op.inputs[0], op.inputs[1], op=op_code, name=name)

@ops.RegisterGradient("EwZXa")
def ew_z_xa_grad(op, dz):
    op_code = op.get_attr("op")
    alpha   = op.get_attr("alpha")
    name    = ew_names[op_code] + "_grad"

    if op_code == NEG_OP:
        return ew_z_xa_op(dz, op=NEG_OP, name=name)

    # use the z values to compute grad for these ops
    # I belive it saves memory with their typical use
    if op_code in (RELU_OP, SIG_OP, TANH_OP):
        return ew_dx_dzza_op(dz, op.outputs[0], op=op_code, alpha=alpha, name=name)

    return ew_dx_dzxa_op(dz, op.inputs[0], op=op_code, alpha=alpha, name=name)

@ops.RegisterGradient("EwZXb")
def ew_z_xb_grad(op, dz):
    op_code = op.get_attr("op")
    name    = ew_names[op_code] + "_grad"

    if op_code == BIASADD_OP:
        return (dz, ew_db_dzb_op(dz, op.inputs[1], op=op_code, name=name))

    if op_code == GAINMUL_OP:
        return ew_dxdg_dzxg_op(dz, op.inputs[0], op.inputs[1], op=op_code, name=name)

    raise ValueError("bad op code")

############################## Filter Infinity/Nans + scale #####################################

filter_tensor_op = _op_module.filter_tensor

# set saturate to 65504.0 to saturate fp16 infinities
def filter_tensor(x, scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False):
    return filter_tensor_op(x, scalar_constant(scale, dtype=tf.float32), saturate=float(saturate), zero_infs=zero_infs, zero_nans=zero_nans)

# alias to filter_tensor that just does scaling by host side scalar value
def scale_tensor(x, scale=1.0):
    return filter_tensor_op(x, scale)

@ops.RegisterGradient("FilterTensor")
def filter_tensor_grad(op, dy):
    return filter_tensor_op(dy, op.inputs[1], saturate=op.get_attr("saturate"), zero_infs=op.get_attr("zero_infs"), zero_nans=op.get_attr("zero_nans")), None

############################## Float Cast #####################################

float_cast_op = _op_module.float_cast

def float_cast(x, dtype, dx_dtype=None, name=None):

    dev = x.op.device.lower()
    if not dev or "cpu" in dev:
        return tf.cast(x, dtype)

    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float16, tf.bfloat16):
        raise ValueError("Only float32 and float16 dtypes supported.")
    # no-op
    if dtype == x.dtype.base_dtype:
        # no-op
        return x
    # default x dtype for dx
    if dx_dtype is None:
        dx_dtype = x.dtype.base_dtype

    return float_cast_op(x, TY=dtype, dx_dtype=dx_dtype, name=name)

@ops.RegisterGradient("FloatCast")
def float_cast_grad(op, dz):
    dx_dtype = op.get_attr("dx_dtype")
    # passthrough
    if dz.dtype == dx_dtype:
        return dz

    return float_cast_op(dz, TY=dx_dtype, dx_dtype=dx_dtype)


############################## Dropout #####################################


gen_dropout_mask_op   = _op_module.gen_dropout_mask
apply_dropout_mask_op = _op_module.apply_dropout_mask


def dropout(x, keep_prob, mask=None, mask_shape=None):

    keep_prob = scalar_constant(keep_prob)

    if mask is None:

        if mask_shape is not None and len(mask_shape) > 0:
            size = 1
            for m_dim, x_dim in zip(mask_shape, x.shape.as_list()):
                # we don't currently support placeholder dims when broadcasting the dropout mask
                assert m_dim == 1 or m_dim == x_dim, f"incompatible mask_shape: {mask_shape} x.shape: {x.shape}"
                size *= m_dim
        else:
            size = 0

        mask = gen_dropout_mask_op(x, get_entropy(), keep_prob, size=size)

    if mask_shape is None:
        mask_shape = []

    return apply_dropout_mask_op(x, mask, keep_prob, mask_shape=mask_shape), mask

@ops.RegisterGradient("ApplyDropoutMask")
def dropout_grad(op, dy):
    mask_shape = op.get_attr("mask_shape")

    dx = apply_dropout_mask_op(dy, op.inputs[1], op.inputs[2], mask_shape=mask_shape)

    return dx, None, None

############################## Concrete Gate for L0 Norm Pruning #####################################

concrete_gate_op       = _op_module.concrete_gate
concrete_gate_grad_op  = _op_module.concrete_gate_grad
concrete_gate_infer_op = _op_module.concrete_gate_infer

def concrete_gate(loga, tempurature=2.0/3.0, limit_a=-0.1, limit_b=1.1, epsilon=1e-6):

    gate, _ = concrete_gate_op(loga, get_entropy(), scalar_constant(tempurature, dtype=tf.float32), limit_a=limit_a, limit_b=limit_b, epsilon=epsilon)
    return gate

def concrete_gate_infer(loga, limit_a=-0.1, limit_b=1.1):
    return concrete_gate_infer_op(loga, limit_a=limit_a, limit_b=limit_b)

@ops.RegisterGradient("ConcreteGate")
def concrete_gate_grad(op, dg, _):
    limit_a = op.get_attr("limit_a")
    limit_b = op.get_attr("limit_b")

    dloga = concrete_gate_grad_op(dg, op.outputs[1], op.inputs[2], limit_a=limit_a, limit_b=limit_b)

    return dloga, None, None


############################## add_n8 #####################################

add_n8_op = _op_module.add_n8

def add_n8(xs, name="AddN"):
    if name is None: name = "AddN"
    return add_n8_op(xs, name=name)

def add_n(xs, name="AddN"):

    if len(xs) == 1:
        return xs[0]

    if name is None: name = "AddN"

    if len(xs) == 2:
        return ew_z_xy_op(xs[0], xs[1], op=0, name=name)

    total = None
    while len(xs):
      xs8 = [] if total is None else [total]
      while len(xs) and len(xs8) < 8:
        xs8.append(xs.pop())
      total = add_n8_op(xs8, name=name)
    return total

old_add_n = None
def replace_add_n():
    from tensorflow.python.ops import math_ops
    global old_add_n
    old_add_n = math_ops.add_n
    math_ops.add_n = add_n

def restore_add_n():
    from tensorflow.python.ops import math_ops
    global old_add_n
    math_ops.add_n = old_add_n


############################## BiasRelu #####################################

bias_relu_op      = _op_module.bias_relu
bias_relu_grad_op = _op_module.bias_relu_grad
bias_grad_op      = _op_module.bias_grad

def bias_relu(x, b, axis=-1, relu=False, fast_gelu=False, atomics=True, bench=0, use_tf=False):

    if relu and fast_gelu:
        raise ValueError("relu and fast_gelu can not both be enabled.")

    dev = x.op.device.lower()
    if use_tf or not dev or "cpu" in dev:
        if b.shape.ndims > 1:
            y = x + b
        else:
            y = tf.nn.bias_add(x, b)
        if relu:
            y = tf.nn.relu(y)
        elif fast_gelu:
            y = y * tf.nn.sigmoid(1.702 * y)

        return y

    relu = 1 if relu else (2 if fast_gelu else 0)

    return bias_relu_op(x, b, axis=axis, relu=relu, bench=bench, atomics=atomics)

@ops.RegisterGradient("BiasRelu")
def bias_relu_grad(op, dy):

    axis    = op.get_attr("axis")
    relu    = op.get_attr("relu")
    atomics = op.get_attr("atomics")
    bench   = op.get_attr("bench")

    if relu:
        x_or_y = op.outputs[0] if relu == 1 else op.inputs[0]
        dx, db, _ = bias_relu_grad_op(dy, x_or_y, op.inputs[1], axis=axis, relu=relu, atomics=atomics, bench=bench)
        return dx, db

    db, _ = bias_grad_op(dy, op.inputs[1], axis=axis, atomics=atomics, bench=bench)

    return (dy, db)

############################## FancyGather #####################################

fancy_gather_op      = _op_module.fancy_gather
fancy_gather_grad_op = _op_module.fancy_gather_grad

def fancy_gather(x, idx, use_tf=False):

    x_rank = len(x.shape)
    i_rank = len(idx.shape)
    assert x_rank > i_rank

    dev = x.device.lower()
    if use_tf or not dev or "cpu" in dev:
        idx = tf.maximum(idx, 0)
        flat_shape = tf.concat([[-1], tf.shape(x)[i_rank + 1:]], axis=0)

        xx = tf.reshape(x, flat_shape)
        ii = tf.expand_dims(
                tf.range(0, tf.reduce_prod(tf.shape(x)[:i_rank])) * tf.shape(x)[i_rank] + tf.reshape(idx, [-1]),
                1)
        return tf.reshape(
            tf.gather_nd(xx, ii),
            tf.concat([tf.shape(idx), tf.shape(x)[i_rank + 1:]], axis=0),
        )

    if x_rank > i_rank + 1:
        # temp restriction for now... easily fixed
        assert x.shape[i_rank + 1:].num_elements() <= 1024

    return fancy_gather_op(x, idx, idx_dim=x.shape[i_rank].value)

@ops.RegisterGradient("FancyGather")
def fancy_gather_grad(op, dy):
    dx = fancy_gather_grad_op(dy, op.inputs[1], idx_dim=op.get_attr("idx_dim"))
    return (dx, None)


############################## ReduceMax #####################################

reduce_max_op      = _op_module.reduce_max
reduce_max_grad_op = _op_module.reduce_max_grad

def reduce_max(x, axis, keepdims=False, use_tf=False):

    shape = x.shape.as_list()
    assert type(axis) is int, "reshape prior to op to support contiguous index ranges"
    assert shape[axis] is not None, "reduction over unknown dimension size not supported"

    if axis < 0:
        axis += len(shape)

    dev = x.op.device.lower()
    if use_tf or not dev or "cpu" in dev or axis == len(shape)-1:
        return tf.reduce_max(x, axis=axis, keepdims=keepdims)

    idx_dtype = tf.uint16 if shape[axis] > 256 else tf.uint8

    y, a = reduce_max_op(x, axis=axis, keepdims=keepdims, TA=idx_dtype)
    return y

@ops.RegisterGradient("ReduceMax")
def reduce_max_grad(op, dy, a):

    axis      = op.get_attr("axis")
    keepdims  = op.get_attr("keepdims")
    axis_size = op.inputs[0].shape[axis].value

    return reduce_max_grad_op(dy, op.outputs[1], axis=axis, axis_size=axis_size, keepdims=keepdims)

############################## AssignAdd #####################################

def assign_add(y, x, name=None):
    return _op_module.assign_add_op(y, x, name=name)


# if mpi_rank == 0:
#     with tf.device("/gpu:0"), tf.name_scope("LogStats"):
#         for i, (grad, param) in enumerate(zip(grads, params)):
#             name = param.op.name + "_" + "_".join(str(x) for x in param.shape.as_list())
#             grads[i] = ew.log_stats(grad, step, logfile="scale_14.txt", name=name)

# ebits = 4
# fbits = 3
# ebias = 8
# for exp in range(1 << ebits):
#     for frac in range(1 << fbits):
#         frac /= 1 << fbits
#         f8 = (1 + frac) * 2**(exp - ebias)
#         l8 = 2**(exp + frac - ebias)
#         print("%2d %.3f %9.5f %9.5f" % (exp-ebias, frac, f8, l8))

