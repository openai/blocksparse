
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, function

from blocksparse.utils import _op_module, scalar_constant

embedding_lookup_op      = _op_module.embedding_lookup
embedding_lookup_grad_op = _op_module.embedding_lookup_grad
float_cast_op            = _op_module.float_cast


def embedding_lookup(emb, idx, sort_grad=True, bench=0, use_tf=False):

    dev = emb.op.device.lower()
    if use_tf or not dev or "cpu" in dev:
        #print("######################### Using TF embeding:", dev)
        y = tf.nn.embedding_lookup(convert_gradient_to_tensor(emb), idx)
    else:
        y = embedding_lookup_op(emb, idx, scalar_constant(emb.shape[0].value, dtype=tf.int32), sorted=sort_grad, bench=bench)
    return y

@ops.RegisterGradient("EmbeddingLookup")
def embedding_lookup_grad(op, dy):
    sort  = op.get_attr("sorted")
    bench = op.get_attr("bench")
    dw = embedding_lookup_grad_op(dy, op.inputs[1], op.inputs[2], sorted=sort, bench=bench)
    if dy.dtype is not tf.float32:
       dw = float_cast_op(dw, TY=dy.dtype, dx_dtype=dy.dtype)

    return dw, None, None

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].shape])
def convert_gradient_to_tensor(x):
    return x