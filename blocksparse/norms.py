
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from blocksparse.utils import _op_module, reduce_mul

layer_norm_op            = _op_module.layer_norm
layer_norm_grad_op       = _op_module.layer_norm_grad
batch_norm_inf_ncdhw_op  = _op_module.batch_norm_inference_ncdhw
batch_norm_ncdhw_op      = _op_module.batch_norm_ncdhw
batch_norm_grad_ncdhw_op = _op_module.batch_norm_grad_ncdhw



def layer_norm(x, g, b, axis=1, segments=1, epsilon=1e-6, relu=False, atomics=True, bench=0, use_tf=False):

    dev = g.op.device.lower()
    if use_tf or not dev or "cpu" in dev:

        if axis < 0:
            axis += len(x.shape)

        K = x.shape[axis].value
        assert g.shape.num_elements() == K
        assert b.shape.num_elements() == K
        assert K % segments == 0
        assert axis != 0 or segments == 1, "Segments only implemented on axis=1 for now"
        K //= segments

        ys = list()
        for s in range(segments):

            segK = slice(s*K, s*K+K)
            segX = [segK if d == axis else slice(None) for d in range(x.shape.ndims)]

            mean, var = tf.nn.moments(x[segX], [axis], keep_dims=True)
            # mean = tf.reduce_mean(x[segX], axis=[axis], keepdims=True)
            # var  = tf.reduce_mean(tf.square(x[segX] - mean), axis=[axis], keepdims=True)
            norm = (x[segX] - mean) * tf.rsqrt(var + epsilon)
            ys.append(norm * g[segK] + b[segK])

        y = tf.concat(ys, axis) if segments > 1 else ys[0]
        if relu:
            y = tf.nn.relu(y)
    else:
        y, m, v, _, _ = layer_norm_op(x, g, b, S=segments, axis=axis, epsilon=epsilon, relu=relu, atomics=atomics, bench=bench)

    return y

@ops.RegisterGradient("LayerNorm")
def layer_norm_grad(op, dy, mean, rstd, p1, p2):
    S       = op.get_attr("S")
    epsilon = op.get_attr("epsilon")
    relu    = op.get_attr("relu")
    axis    = op.get_attr("axis")
    atomics = op.get_attr("atomics")
    bench   = op.get_attr("bench")
    dx, dg, db, _, _ = layer_norm_grad_op(dy, op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[1], op.outputs[2], S=S, axis=axis, epsilon=epsilon, relu=relu, atomics=atomics, bench=bench)
    return dx, dg, db

def batch_norm_inference(x, g, b, m, v, epsilon=1e-6):
    shape = x.shape
    C     = int(shape[1])
    DHW   = int(shape[2:].num_elements())
    assert g.get_shape().num_elements() == C
    assert b.get_shape().num_elements() == C
    assert m.get_shape().num_elements() == C
    assert v.get_shape().num_elements() == C
    return batch_norm_inf_ncdhw_op(x, g, b, m, v, DHW=DHW, eps=epsilon)

@ops.RegisterGradient("BatchNormInferenceNCDHW")
def batch_norm_inf_grad(op, dy):
    return (dy, None, None, None, None)


def batch_norm(x, g, b, epsilon=1e-6):
    shape = x.shape
    C     = int(shape[1])
    DHW   = int(shape[2:].num_elements())
    magic = _magic64u(DHW)
    assert g.get_shape().num_elements() == C
    assert b.get_shape().num_elements() == C
    return batch_norm_ncdhw_op(x, g, b, DHW=DHW, magic_DHW=magic[0], shift_DHW=magic[1], eps=epsilon)

@ops.RegisterGradient("BatchNormNCDHW")
def batch_norm_grad(op, dy, mean, var):
    eps = op.get_attr("eps")
    DHW = op.get_attr("DHW")
    magic_DHW = op.get_attr("magic_DHW")
    shift_DHW = op.get_attr("shift_DHW")
    return batch_norm_grad_ncdhw_op(dy, op.inputs[0], op.inputs[1], op.outputs[1], op.outputs[2], DHW=DHW, magic_DHW=magic_DHW, shift_DHW=shift_DHW, eps=eps)



def layer_norm_test(x, g, b, axis=1, segments=1, epsilon=1e-6, relu=False):

    x_shape = x.shape
    K = x_shape[axis]
    if axis == 0:
        x = x.reshape(K,-1)
        g = g.reshape(K, 1)
        b = b.reshape(K, 1)
    else:
        axis = 1
        x = x.reshape(-1, K)
        g = g.reshape( 1, K)
        b = b.reshape( 1, K)
    K //= segments

    y = np.empty_like(x)
    for s in range(segments):

        segK = slice(s*K, s*K+K)
        seg  = (segK, slice(None)) if axis == 0 else (slice(None), segK)

        mean = np.mean(x[seg], axis=axis, keepdims=True)
        var  = np.var(x[seg], axis=axis, keepdims=True)
        rstd = np.reciprocal(np.sqrt(var + epsilon))
        xhat = (x[seg] - mean) * rstd

        y[seg] = xhat*g[seg] + b[seg]
        if relu:
            y[seg] = np.maximum(y[seg], 0.0)

    return y.reshape(x_shape)

def layer_norm_grad_test(dy, x, g, b, axis=1, segments=1, epsilon=1e-6, relu=False):

    x_shape = x.shape
    K = x_shape[axis]
    if axis == 0:
        dy = dy.reshape(K,-1)
        x  =  x.reshape(K,-1)
        g  =  g.reshape(K, 1)
        b  =  b.reshape(K, 1)
    else:
        axis = 1
        dy = dy.reshape(-1, K)
        x  =  x.reshape(-1, K)
        g  =  g.reshape( 1, K)
        b  =  b.reshape( 1, K)
    K //= segments

    dy = dy.copy()
    dx = np.empty_like(dy)
    dg = np.empty_like(g)
    db = np.empty_like(b)
    for s in range(segments):

        segK  = slice(s*K, s*K+K)
        seg   = (segK, slice(None)) if axis == 0 else (slice(None), segK)

        mean  = np.mean(x[seg], axis=axis, keepdims=True)
        xmean = x[seg] - mean
        xvar  = np.var(x[seg], axis=axis, keepdims=True)
        xstdr = np.reciprocal(np.sqrt(xvar + epsilon))
        xhat  = xmean * xstdr

        if relu:
            dy[seg] = dy[seg] * ((xhat*g[seg] + b[seg]) > 0.0)

        #print("x:%.2f, mean:%.2f, rstd:%.2f, xhat:%.2f, dy:%.2f\n" % (x[0,0], mean[0,0], xstdr[0,0], xhat[0,0], dy[0,0]));

        dg[seg] = np.sum(dy[seg] * xhat, axis=1-axis, keepdims=True)
        db[seg] = np.sum(dy[seg],        axis=1-axis, keepdims=True)
        dy[seg] = dy[seg] * g[seg]

        sum1 = np.sum(xhat * dy[seg], axis=axis, keepdims=True)
        sum2 = np.sum(dy[seg],        axis=axis, keepdims=True)
        dx[seg]   = (dy[seg] - ((xhat * sum1 + sum2) / float(K))) * xstdr

    return dx.reshape(x_shape), dg, db


def batch_norm_inf_test(x, g, b, m, v, epsilon=1e-6):

    xshape = x.shape
    N = xshape[0]
    C = xshape[1]
    x = x.reshape(N, C,-1)
    g = g.reshape(1, C, 1)
    b = b.reshape(1, C, 1)
    m = m.reshape(1, C, 1)
    v = v.reshape(1, C, 1)

    rstd = np.reciprocal(np.sqrt(v + epsilon))
    xhat = (x - m) * rstd

    return (xhat*g + b).reshape(xshape)

def batch_norm_test(x, g, b, epsilon=1e-6):

    xshape = x.shape
    N = xshape[0]
    C = xshape[1]
    x = x.reshape(N, C,-1)
    g = g.reshape(1, C, 1)
    b = b.reshape(1, C, 1)

    mean = np.mean(x, axis=(0,2), keepdims=True)
    var  = np.var (x, axis=(0,2), keepdims=True)
    rstd = np.reciprocal(np.sqrt(var + epsilon))
    xhat = (x - mean) * rstd

    return (xhat*g + b).reshape(xshape), mean.reshape(C), var.reshape(C)

def batch_norm_grad_test(dy, x, g, m, v, epsilon=1e-6):

    xshape = x.shape
    N = xshape[0]
    C = xshape[1]
    rNDHW  = 1.0 / reduce_mul(xshape[2:], N)

    dy = dy.reshape(N, C,-1)
    x  =  x.reshape(N, C,-1)
    g  =  g.reshape(1, C, 1)
    m  =  m.reshape(1, C, 1)
    v  =  v.reshape(1, C, 1)

    rstd = np.reciprocal(np.sqrt(v + epsilon))
    xhat = (x - m) * rstd;

    dg = np.sum(dy * xhat, axis=(0,2), keepdims=True)
    db = np.sum(dy,        axis=(0,2), keepdims=True)
    z  = (xhat * dg + db) * rNDHW;
    dx = (dy - z) * rstd * g;

    return dx.reshape(xshape), dg.reshape(C), db.reshape(C)




# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def _magic32u(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find magic number for division")


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def _magic64u(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = _magic32u(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

# for d in range(1,1000000):
#     magic, shift = _magic32u(0x7fffffff, d)
#     if shift < 32 or len(hex(magic)) > 10:
#         if magic != 1:
#             print(d, magic, shift)

