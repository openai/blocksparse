
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))

############################## Top-K #####################################


top_k_op                = _op_module.topk
rectified_top_k_op      = _op_module.rectified_top_k
masked_softmax_op       = _op_module.masked_softmax
masked_top_k_softmax_op = _op_module.masked_top_k_softmax
masked_softmax_grad_op  = _op_module.masked_softmax_grad
ew_dx_dzza_op           = _op_module.ew_dx_dzza

def top_k(x, k):
    assert k <= x.shape[-1].val <= 1024
    return top_k_op(x, k)

def rectified_top_k(x, k, rebase=True):
    assert k <= x.shape[-1].value <= 1024
    return rectified_top_k_op(x, k, rebase=rebase)

@ops.RegisterGradient("RectifiedTopK")
def rectified_top_k_grad(op, dz):
    # same grad as relu
    return ew_dx_dzza_op(dz, op.outputs[0], op=RELU_OP)

@ops.RegisterGradient("Topk")
def top_k_grad(op, grad, _):

  in_shape  = array_ops.shape(op.inputs[0])
  ind_shape = array_ops.shape(op.outputs[1])

  ind_lastdim = array_ops.gather(ind_shape, array_ops.size(ind_shape) - 1)
  # Flatten indices to 2D.
  ind_2d = array_ops.reshape(op.outputs[1], array_ops.stack([-1, ind_lastdim]))

  in_lastdim = array_ops.gather(in_shape, array_ops.size(in_shape) - 1)
  outerdim   = array_ops.shape(ind_2d)[0]
  # Compute linear indices (flattened to 1D).
  ind = array_ops.reshape(ind_2d + array_ops.expand_dims(
      math_ops.range(0, outerdim * in_lastdim, in_lastdim), -1), [-1])

  # Substitute grad to appropriate locations and fill the rest with zeros,
  # finally reshaping it to the original input shape.
  return [
      array_ops.reshape(
          sparse_ops.sparse_to_dense(
              ind,
              array_ops.reshape(math_ops.reduce_prod(in_shape), [1]),
              array_ops.reshape(grad, [-1]),
              validate_indices=False), in_shape),
      array_ops.zeros([], dtype=dtypes.int32)
  ]


def rectified_top_k_test(x, k, rebase=True):

    a = np.argsort(x)[:,::-1]
    y = np.zeros(x.shape, dtype=np.float32)
    for i in range(x.shape[0]):

        # get min value among topk
        base = max(x[i,a[i,k-1]], 0.0) if rebase else 0.0
        #print(base, a[i,k-1])

        # write just the topk values from x to y
        y[i,a[i,:k]] = np.maximum(x[i,a[i,:k]], base) - base

    return y


def masked_top_k_softmax(x, k, mask=None, scale=1.0):

    assert k <= x.shape[-1].value <= 1024

    if mask is not None:
        x_shape = x.shape.as_list()
        m_shape = mask.shape.as_list()

        assert len(x_shape) == len(m_shape)
        for i in range(len(m_shape)):
            assert m_shape[i] in (1, x_shape[i])
        mask = [ mask ]
    else:
        mask = []

    return masked_top_k_softmax_op(x, k, scale, mask)


def softmax(x, scale=1.0, bench=0):
    return masked_softmax_op(x, scale, [], bench=bench)

def masked_softmax(x, mask=None, scale=1.0, bench=0):
    if mask is not None:
        x_shape = x.shape.as_list()
        m_shape = mask.shape.as_list()

        assert len(x_shape) == len(m_shape)
        for i in range(len(m_shape)):
            assert m_shape[i] in (1, x_shape[i])
        mask = [ mask ]
    else:
        mask = []

    return masked_softmax_op(x, scale, mask, bench=bench)


@ops.RegisterGradient("MaskedTopKSoftmax")
def masked_top_k_softmax_grad(op, dy):

    n_mask = op.get_attr("n_mask")
    mask   = [ op.inputs[3] ] if n_mask else []
    dx = masked_softmax_grad_op(dy, op.outputs[0], op.inputs[2], mask)
    if n_mask:
        return (dx, None, None, None)
    return (dx, None, None)

@ops.RegisterGradient("MaskedSoftmax")
def masked_softmax_grad(op, dy):

    bench  = op.get_attr("bench")
    n_mask = op.get_attr("n_mask")
    mask   = [ op.inputs[2] ] if n_mask else []
    dx = masked_softmax_grad_op(dy, op.outputs[0], op.inputs[1], mask, bench=bench)
    if n_mask:
        return (dx, None, None)
    return (dx, None)

def masked_softmax_test(x, mask=None, scale=1.0):
    x_shape = x.shape

    if mask is not None:
        x = x.reshape(-1, mask.size)
        y = np.empty(x.shape, dtype=np.float32)
        y.fill(-np.finfo(np.float32).max)
        nz = np.nonzero(mask.reshape(-1))
        y[:,nz] = x[:,nz] * mask.reshape(1,-1)[:,nz] * scale
    else:
        y = x * scale

    y = y.reshape(-1, x_shape[-1])
    m = np.max(y, axis=1, keepdims=True)
    z = np.exp(y - m) / np.sum(np.exp(y - m), axis=1, keepdims=True)

    return z.reshape(x_shape)

def masked_top_k_softmax_test(x, k, mask=None, scale=1.0):

    x_shape = x.shape

    if mask is not None:
        x = x.reshape(-1, mask.size)
        y = np.empty(x.shape, dtype=np.float32)
        y.fill(-np.finfo(np.float32).max)
        nz = np.nonzero(mask.reshape(-1))
        y[:,nz] = x[:,nz] * mask.reshape(1,-1)[:,nz] * scale
    else:
        y = x * scale

    y = y.reshape(-1, x_shape[-1])
    a = np.argsort(y)[:,::-1]
    z = np.zeros(y.shape, dtype=np.float32)
    for i in range(y.shape[0]):
        # get max value among top_k
        max_val = y[i,a[i,0]]
        # compute softmax on just the top_k values
        z[i,a[i,:k]] = np.exp(y[i,a[i,:k]] - max_val) / np.sum(np.exp(y[i,a[i,:k]] - max_val))

    return z.reshape(x_shape)

def masked_softmax_grad_test(dy, y, mask=None, scale=1.0):

    if mask is None:
        mask = 1.0

    return (dy - np.sum(dy * y, axis=-1, keepdims=True)) * y * mask * scale

# m = np.zeros((10,10), dtype=np.float32)
# for y, x in np.ndindex(m.shape):
#     if x <= y: m[y,x] = 1.0
# x = np.arange(1,101, dtype=np.float32).reshape(1,10,10)
# y = masked_top_k_softmax_test(x, 5, mask=m)

############################## Transpose 0213 #####################################

transpose_0213_op = _op_module.transpose0213

def transpose_0213(x):
    return transpose_0213_op(x)

@ops.RegisterGradient("Transpose0213")
def transpose_0213_grad(op, dy):
    return transpose_0213_op(dy)

