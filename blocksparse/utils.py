
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import numpy as np
import tensorflow as tf
from operator import mul
if sys.version_info >= (3, 0):
    from functools import reduce

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))
# for x in dir(_op_module):
#     print(x)
# exit()

g_entropy = None
# max of 80 SMs currently for V100
# 3 lfsr's, 1024 max threads per SM
entropy_size = 80*3*1024

def set_entropy(init=None):
    global g_entropy
    if init is None:
        init = np.random.randint(-(1<<31), (1<<31), size=entropy_size, dtype=np.int32).view(np.float32)
    with tf.device("/gpu:0"):
        g_entropy = tf.get_variable("Entropy", initializer=init, trainable=False)

def get_entropy():
    global g_entropy
    if g_entropy is None:
        # we could call it here for you but if the Session is created more than once
        # then g_entropy needs to be created again.
        raise ValueError("Call bs.set_entropy() after creating Session, then init global variables.")
    return g_entropy

g_scalar_const_cache = dict()

def scalar_constant(value, dtype=None, name=None):

    if isinstance(value, tf.Tensor):
        return value

    if not isinstance(value, (int, float)):
        raise ValueError("Not a scalar value.")

    if isinstance(value, np.float):
        value = float(value)
    elif isinstance(value, np.int):
        value = int(value)

    global g_scalar_const_cache

    if value not in g_scalar_const_cache:
        g_scalar_const_cache[value] = list()

    default_graph = tf.get_default_graph()
    for tf_const in g_scalar_const_cache[value]:
        # constants are tied to the (sub)graph
        if tf_const.graph is default_graph:
            return tf_const

    with tf.device("/cpu:0"), tf.control_dependencies(None):
        tf_const = tf.constant(value, dtype=dtype, name=name)
    g_scalar_const_cache[value].append(tf_const)
    return tf_const

def reset_scalar_constants():
    global g_scalar_const_cache
    g_scalar_const_cache = dict()


def is_param_casted(param):
    for c in param.op.outputs[0].consumers():
        if c.type == "Identity" and c.name[-4:] == "read":
            consumers = c.outputs[0].consumers()
            # should just be 1 cast op, but maybe allow for more creative uses
            if len(consumers) <= 5:
                for op in consumers:
                    if "Cast" in op.type:
                        return True
    return False


def reduce_mul(vals, init=1):
    return reduce(mul, vals, init)

def ceil_div(x, y):
    return -(-x // y)

def z_order_2d(x, y):
    answer = 0
    bits = max(len(bin(x)), len(bin(y))) - 2
    for i in range(bits):
        mshifted = 1 << i;
        shift = i
        answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1))
        #print mshifted, shift, answer, bin(answer)
    return answer

# Morton ordering (z-order) of 3D coords
def z_order_3d(z, y, x):
    answer = 0
    bits = max(len(bin(x)), len(bin(y)), len(bin(z))) - 2
    for i in range(bits):
        mshifted = 1 << i;
        shift = i << 1
        answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1)) | ((z & mshifted) << (shift + 2))
        #print mshifted, shift, answer, bin(answer)
    return answer

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def magic32u(nmax, d):
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
def magic64u(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = magic32u(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)

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


def dilation_size(S, dil=1):
    return S * dil - dil + 1

def out_dim(S, W, pad, std=1, dil=1):
    return ceil_div(W - dilation_size(S, dil) + 1 + 2*pad, std)
    #return ((W - dilation_size(S, dil) + 2 * pad) // std) + 1

def same_pad(S, dil=1):
    return dilation_size(S, dil) // 2

def backward_pad(S, pad, dil=1):
    return dilation_size(S, dil) - pad - 1

def conv_slice(q, W, S, pad, std=1, dil=1):
    qs = q * std - pad
    ws = list()
    for s in range(S):
        w = qs + s * dil
        if w >= 0 and w < W:
            ws.append(w)
    return ws

def deconv_slice(x, Q, S, bpad, std=1, dil=1):
    xs = x - bpad
    e = list()
    for s in range(S):
        q = xs + s * dil
        if q % std == 0:
            q //= std
            if q >= 0 and q < Q:
                e.append(q)
    return e

def bst_conv_layout(input_h=1, input_w=1, filter_h=1, filter_w=1, stride=1, blk_size=32, autoregressive=True):

    H = input_h
    W = input_w
    R = filter_h
    S = filter_w

    assert H % stride == 0 or H == 1
    assert W % stride == 0

    P = H // stride or 1
    Q = W // stride

    if H == 1:
        R = 1
        pad_r = 0
    else:
        pad_r = -1
        for r in range(R):
            if P == out_dim(R, H, r, stride):
                pad_r = r
                break
    assert pad_r >= 0, "Even size filters only work with stride 2."

    pad_s = -1
    for s in range(S):
        if Q == out_dim(S, W, s, stride):
            pad_s = s
            break
    assert pad_s >= 0, "Even size filters only work with stride 2."

    print(f"P:{P} Q:{Q} H:{H} W:{W} R:{R} S:{S} std:{stride} pad_r:{pad_r} pad_s:{pad_s}")

    assert P*Q % blk_size == 0, f"P:{P} Q:{Q}"
    assert H*W % blk_size == 0, f"H:{H} W:{W}"

    mask_set = set()
    layout = np.zeros((P*Q//blk_size, H*W//blk_size), dtype=np.bool)

    # just compute the output pixels within the tile
    for p, q in np.ndindex(P, Q):
        for     h in conv_slice(p, H, R, pad_r, stride):
            for w in conv_slice(q, W, S, pad_s, stride):
                x = h*W + w
                y = p*Q + q
                if not autoregressive or p*stride*Q*stride + q*stride >= x:
                    layout[y//blk_size, x//blk_size] = 1
                    mask_set.add((y, x))

    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.zeros(blk_shape, dtype=np.bool)

        q0 = qry_idx*blk_shape[0]
        k0 = key_idx*blk_shape[1]
        for q, k in np.ndindex(blk_shape):
            if (q0 + q, k0 + k) in mask_set:
                mask[q, k] = 1
        return mask

    return layout, cb

# layout, cb = bst_conv_layout(input_h=64, input_w=64, filter_h=15, filter_w=15, stride=1, blk_size=8)
# layout, cb = bst_conv_layout(input_h=64, input_w=64, filter_h=15, filter_w=15, stride=2, blk_size=8)
# layout, cb = bst_conv_layout(input_h=64, input_w=64, filter_h= 8, filter_w= 8, stride=2, blk_size=8)

# layout, cb = bst_conv_layout(input_w=1024, filter_w=225, stride=1, blk_size=8)
# layout, cb = bst_conv_layout(input_w=1024, filter_w=225, stride=2, blk_size=8)
# layout, cb = bst_conv_layout(input_w=1024, filter_w=256, stride=2, blk_size=8)

# np.savetxt("layout.txt", layout, fmt="%d")
# exit()

def bst_deconv_layout(output_h=1, output_w=1, filter_h=1, filter_w=1, stride=1, blk_size=32, autoregressive=True):

    H = output_h
    W = output_w
    R = filter_h
    S = filter_w

    assert H % stride == 0 or H == 1
    assert W % stride == 0

    P = H // stride or 1
    Q = W // stride

    if H == 1:
        R = 1
        pad_r = 0
    else:
        pad_r = -1
        for r in range(R):
            if P == out_dim(R, H, r, stride):
                pad_r = backward_pad(R,r)
                break
    assert pad_r >= 0, "Even size filters only work with stride 2."

    pad_s = -1
    for s in range(S):
        if Q == out_dim(S, W, s, stride):
            pad_s = backward_pad(S,s)
            break
    assert pad_s >= 0, "Even size filters only work with stride 2."

    print(f"P:{P} Q:{Q} H:{H} W:{W} R:{R} S:{S} std:{stride} pad_r:{pad_r} pad_s:{pad_s}")

    assert P*Q % blk_size == 0, f"P:{P} Q:{Q}"
    assert H*W % blk_size == 0, f"H:{H} W:{W}"

    mask_set = set()
    layout = np.zeros((H*W//blk_size, P*Q//blk_size), dtype=np.bool)

    # just compute the output pixels within the tile
    for h, w in np.ndindex(H, W):
        for     p in deconv_slice(h, P, R, pad_r, stride):
            for q in deconv_slice(w, Q, S, pad_s, stride):
                y = h*W + w
                x = p*Q + q
                if not autoregressive or y >= p*stride*Q*stride + q*stride:
                    layout[y//blk_size, x//blk_size] = 1
                    mask_set.add((y, x))

    def cb(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.zeros(blk_shape, dtype=np.bool)

        q0 = qry_idx*blk_shape[0]
        k0 = key_idx*blk_shape[1]
        for q, k in np.ndindex(blk_shape):
            if (q0 + q, k0 + k) in mask_set:
                mask[q, k] = 1
        return mask

    return layout, cb

# layout, cb = bst_deconv_layout(output_h=64, output_w=64, filter_h=15, filter_w=15, stride=1, blk_size=8)
# layout, cb = bst_deconv_layout(output_h=64, output_w=64, filter_h=15, filter_w=15, stride=2, blk_size=8)
# layout, cb = bst_deconv_layout(output_h=64, output_w=64, filter_h= 8, filter_w= 8, stride=2, blk_size=8)

# layout, cb = bst_deconv_layout(output_w=1024, filter_w=225, stride=1, blk_size=8)
# layout, cb = bst_deconv_layout(output_w=1024, filter_w=225, stride=2, blk_size=8)
# layout, cb = bst_deconv_layout(output_w=1024, filter_w=256, stride=2, blk_size=8)

# np.savetxt("layout.txt", cb((8,8), 0, 0, 0, 0), fmt="%d")
