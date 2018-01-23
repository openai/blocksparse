
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import numpy as np
import tensorflow as tf
import operator
from tensorflow.python.framework import ops
if sys.version_info >= (3, 0):
    from functools import reduce

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

############################## fused_lstm_gates #####################################

lstm_gates_op       = _op_module.lstm_gates
lstm_gates_grad_op  = _op_module.lstm_gates_grad
lstm_gates4_op      = _op_module.lstm_gates4
lstm_gates4_grad_op = _op_module.lstm_gates4_grad

def fused_lstm_gates(c, *args, **kwargs):
    # returns c_next, h_next
    
    assert len(kwargs) <= 1
    name = kwargs.pop('name', None)    
    # args is h (all four gates fused in single tensor)
    if len(args) == 1:
        return lstm_gates_op(c, args[0], name=name)
    # args is i, f, o, u
    assert len(args) == 4
    return lstm_gates4_op(c, *args, name=name)

@ops.RegisterGradient("LSTMGates")
def fused_lstm_gates_grad(op, ec, eh):
    # returns dc, dh

    # in mixed precision mode tf will send the wrong dtype for the first "ec"
    # in our kernels we just conditionaly load zero instead of reading the constant tensor
    if ec is None or ec.dtype != eh.dtype:
        return lstm_gates_grad_op(op.inputs[0], op.inputs[1], [eh] )

    return lstm_gates_grad_op(op.inputs[0], op.inputs[1], [eh, ec] )

@ops.RegisterGradient("LSTMGates4")
def fused_lstm_gates4_grad(op, ec, eh):
    # returns dc, dh

    # in mixed precision mode tf will send the wrong dtype for the first "ec"
    # in our kernels we just conditionaly load zero instead of reading the constant tensor
    if ec is None or ec.dtype != eh.dtype:
        return lstm_gates4_grad_op(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], [eh] )

    return lstm_gates4_grad_op(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], [eh, ec] )

############################## Split4 #####################################

split4_op  = _op_module.split4
concat4_op = _op_module.concat4

def split4(x):
    return split4_op(x)

def concat4(x0, x1, x2, x3):
    return concat4_op(x0, x1, x2, x3)

@ops.RegisterGradient("Split4")
def split4_grad(op, dz0, dz1, dz2, dz3):
    return concat4_op(dz0, dz1, dz2, dz3)

@ops.RegisterGradient("Concat4")
def concat4_grad(op, dz):
    return split4_op(dz)

############################## Float Cast #####################################

float_cast_op = _op_module.float_cast

def float_cast(x, dtype, dx_dtype=None):
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float16, tf.bfloat16):
        raise ValueError("Only float32 and float16 dtypes supported.")
    # no-op
    if dtype == x.dtype.base_dtype:
        # return identity for code depening on an op being here.
        return tf.identity(x)
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

############################## Sparse Relu #####################################


sparse_relu_op = _op_module.sparse_relu

def sparse_relu(x, alpha=1.0):
    return sparse_relu_op(x, alpha)

@ops.RegisterGradient("SparseRelu")
def sparse_relu_grad(op, dz):
    # same grad as relu
    return ew_dx_dzza_op(dz, op.outputs[0], op=RELU_OP)

def sparse_relu_test(x, alpha=1.0):

    mean = np.mean(x, axis=1, keepdims=True)
    std  = np.std(x, axis=1, keepdims=True)
    cutoff = mean + alpha*std
    return np.maximum(x, cutoff) - cutoff


############################## Dropout #####################################


dropout_op      = _op_module.dropout
dropout_mask_op = _op_module.dropout_mask
dropout_grad_op = _op_module.dropout_grad

def dropout(x, keep_prob=0.8, mask=None):
    if mask is None:
        return dropout_op(x, keep_prob=keep_prob)
    return dropout_mask_op(x, mask)

@ops.RegisterGradient("Dropout")
def dropout_grad(op, dy, dm):
    return dropout_grad_op(dy, op.outputs[1])

@ops.RegisterGradient("DropoutMask")
def dropout_grad(op, dy):
    dx = dropout_grad_op(dy, op.inputs[1])
    return dx, None


############################## add_n8 #####################################

add_n8_op = _op_module.add_n8

def add_n8(xs, name="AddN"):
    if name is None:
        name = "AddN"
    return add_n8_op(xs, name=name)

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

