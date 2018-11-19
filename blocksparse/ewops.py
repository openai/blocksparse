
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))
# for x in dir(_op_module):
#     print(x)
# exit()

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
BIASADD_OP = 16
GAINMUL_OP = 17

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
    "Biasadd",
    "Gainmul",
]

def broadcast_check(x, y, ew_op, bc_op, tf_op, name):
    xshape = x.get_shape().as_list()
    yshape = y.get_shape().as_list()

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

def elu (x, alpha=1.0, name=None): return ew_z_xa_op(x, op=ELU_OP, alpha=alpha, name=ew_names[ELU_OP] if name is None else name)

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
    return filter_tensor_op(x, scale, saturate=float(saturate), zero_infs=zero_infs, zero_nans=zero_nans)

# alias to filter_tensor that just does scaling by host side scalar value
def scale_tensor(x, scale=1.0):
    return filter_tensor_op(x, scale)

@ops.RegisterGradient("FilterTensor")
def filter_tensor_grad(op, dy):
    return filter_tensor_op(dy, op.inputs[1], saturate=op.get_attr("saturate"), zero_infs=op.get_attr("zero_infs"), zero_nans=op.get_attr("zero_nans")), None

############################## Float Cast #####################################

float_cast_op = _op_module.float_cast

def float_cast(x, dtype, dx_dtype=None):

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

    return float_cast_op(x, TY=dtype, dx_dtype=dx_dtype)

@ops.RegisterGradient("FloatCast")
def float_cast_grad(op, dz):
    dx_dtype = op.get_attr("dx_dtype")
    # passthrough
    if dz.dtype == dx_dtype:
        return dz

    return float_cast_op(dz, TY=dx_dtype, dx_dtype=dx_dtype)


############################## Dropout #####################################


dropout_op      = _op_module.dropout
dropout_mask_op = _op_module.dropout_mask
dropout_grad_op = _op_module.dropout_grad

def dropout(x, keep_prob=None, scale=None, mask=None):

    assert keep_prob is not None or mask is not None

    if keep_prob is None:
        keep_prob = 1.0
    if type(keep_prob) in (float, int):
        keep_prob = tf.constant(float(keep_prob))

    if scale is None:
        scale = tf.reciprocal(keep_prob)
    elif type(scale) in (float, int):
        scale = tf.constant(float(scale))

    if mask is None:
        return dropout_op(x, keep_prob, scale)
    return dropout_mask_op(x, mask, scale)

@ops.RegisterGradient("Dropout")
def dropout_grad(op, dy, dm):
    dx = dropout_grad_op(dy, op.outputs[1], op.inputs[2])
    return dx, None, None

@ops.RegisterGradient("DropoutMask")
def dropout_grad(op, dy):
    dx = dropout_grad_op(dy, op.inputs[1], op.inputs[2])
    return dx, None, None


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

def bias_relu(x, b, relu=True, atomics=True, bench=0, use_tf=False):

    dev = x.op.device.lower()
    if use_tf or not dev or "cpu" in dev:
        y = tf.nn.bias_add(x, b)
        if relu:
            y = tf.nn.relu(y)
        return y

    return bias_relu_op(x, b, relu=relu, bench=bench, atomics=atomics)

@ops.RegisterGradient("BiasRelu")
def bias_relu_grad(op, dy):

    atomics = op.get_attr("atomics")
    bench   = op.get_attr("bench")

    if op.get_attr("relu"):
        #return bias_relu_grad_op(dy, op.outputs[0], op.inputs[1], bench=op.get_attr("bench"))
        dx, db, _ = bias_relu_grad_op(dy, op.outputs[0], op.inputs[1], atomics=atomics, bench=bench)
        return dx, db

    # dx = dy if no relu
    #db = bias_grad_op(dy, op.inputs[1], bench=op.get_attr("bench"))
    db, _ = bias_grad_op(dy, op.inputs[1], atomics=atomics, bench=bench)

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



# if mpi_rank == 0:
#     with tf.device("/gpu:0"), tf.name_scope("LogStats"):
#         for i, (grad, param) in enumerate(zip(grads, params)):
#             name = param.op.name + "_" + "_".join(str(x) for x in param.shape.as_list())
#             grads[i] = ew.log_stats(grad, step, logfile="scale_14.txt", name=name)

# def sig_test(x):
#     return 1.0/(1.0 + np.exp(-x))

# def relu_test(x):
#     return np.maximum(x, 0.0)

# def elu_test(x, a=1.0):
#     return x * (x > 0.0)  +  a * (np.exp(x) - 1.0) * (x <= 0.0)


# def add_grad_test(dz, x, y):
#     return (dz, dz)

# def mul_grad_test(dz, x, y):
#     return (dz*y, dz*x)

# def sub_grad_test(dz, x, y):
#     return (dz, -dz)

# def div_grad_test(dz, x, y):
#     return (dz/y, -dz * x / (y*y))

# def max_grad_test(dz, x, y):
#     return (dz * (x >= y), dz * (y >= x))

# def min_grad_test(dz, x, y):
#     return (dz * (x <= y), dz * (y <= x))


# def sig_grad_test(dz, z):
#     return dz * (z - z*z)

# def tanh_grad_test(dz, z):
#     return dz * (1.0 - z*z)

# def relu_grad_test(dz, z):
#     return dz * (z > 0.0)


# def neg_grad_test(dz, x):
#     return -dz

# def rcp_grad_test(dz, x):
#     return -dz / (x*x)

# def sqr_grad_test(dz, x):
#     return dz * x * 2.0

# def sqrt_grad_test(dz, x):
#     return 0.5 * dz / np.sqrt(x)

# def exp_grad_test(dz, x):
#     return dz * np.exp(x)

# def log_grad_test(dz, x):
#     return dz / x


# def elu_grad_test(dz, x, a=1.0):
#     return dz * (x > 0.0) + dz * (x <= 0.0) *(a * (np.exp(x) - 1.0) + a)


# def bias_add_grad_test(dz, x, b):
#     return (dz, np.sum(dz, axis=0, keepdims=True))

# def gain_mul_grad_test(dz, x, g):
#     return (dz * g, np.sum(dz * x, axis=0, keepdims=True))
# ebits = 4
# fbits = 3
# ebias = 8
# for exp in range(1 << ebits):
#     for frac in range(1 << fbits):
#         frac /= 1 << fbits
#         f8 = (1 + frac) * 2**(exp - ebias)
#         l8 = 2**(exp + frac - ebias)
#         print("%2d %.3f %9.5f %9.5f" % (exp-ebias, frac, f8, l8))

