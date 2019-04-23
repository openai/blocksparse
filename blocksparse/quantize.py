
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from blocksparse.utils import _op_module, get_entropy


############################## Quantization #####################################

quantize_op  = _op_module.quantize
log_stats_op = _op_module.log_stats

class QuantizeSpec(object):
    def __init__(self, ebits=4, fbits=3, emax=None, stochastic=0, denorm=True, frequency=1024, mode=0, bias_pad=2, stdv_mul=4.0, logfile="", copy=None):

        if copy is None:
            if emax is None:
                emax = (1 << (ebits-1)) - 1 # default symetric
            self.ebits     = ebits
            self.fbits     = fbits
            self.emax      = emax
            self.stoch     = stochastic
            self.denorm    = denorm
            self.freq      = frequency
            self.mode      = mode
            self.bias_pad  = bias_pad
            self.stdv_mul  = stdv_mul
            self.logfile   = logfile
        else:
            self.ebits     = copy.ebits
            self.fbits     = copy.fbits
            self.emax      = copy.emax
            self.stoch     = copy.stoch
            self.denorm    = copy.denorm
            self.freq      = copy.freq
            self.mode      = copy.mode
            self.bias_pad  = copy.bias_pad
            self.stdv_mul  = copy.stdv_mul
            self.logfile   = copy.logfile or logfile


log_init = set()
quant_headers = [
    "sat_pct",
    "ftz_pct",
    "exp_max",
    "exp_min",
    "max",
    "mean",
    "stdv",
    "mean+stdv5",
    "max_stat_lo",
    "max_stat_hi",
    "count",
    "name",
]

log_timestamp = None

def get_timestamp():
    global log_timestamp
    if log_timestamp is None:
        log_timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    return log_timestamp


def quantize(x, qspec, b_qspec=None, name=None):

    if name is None:
        name = "quantize"

    if b_qspec is None:
        b_qspec = qspec

    if x.dtype.base_dtype == tf.bfloat16:
        for spec in (qspec, b_qspec):
            assert spec.fbits <= 7, "bfloat only supports up to 7 fractional bits"

    global log_init
    for spec in (qspec, b_qspec):
        if spec.logfile and spec.logfile not in log_init:
            with open(spec.logfile, 'w') as log:
                log.write("\t".join(quant_headers) + "\n")
            log_init.add(spec.logfile)

    e = [get_entropy()] if qspec.stoch == 2 else []

    reuse = tf.get_variable_scope().reuse

    with tf.device("/cpu:0"), tf.variable_scope("quantize"):
        exp_f = tf.get_variable(name + "_exp_f", dtype=tf.int64, initializer=np.int64(qspec.emax),   trainable=False)
        exp_b = tf.get_variable(name + "_exp_b", dtype=tf.int64, initializer=np.int64(b_qspec.emax), trainable=False)

    return quantize_op(x, exp_f, exp_b, e,
        ebits      = qspec.ebits,
        fbits      = qspec.fbits,
        stoch      = qspec.stoch,
        denorm     = qspec.denorm,
        freq       = (not reuse and qspec.freq),
        mode       = qspec.mode,
        bias_pad   = qspec.bias_pad,
        stdv_mul   = qspec.stdv_mul,
        logfile    = qspec.logfile,
        b_ebits    = b_qspec.ebits,
        b_fbits    = b_qspec.fbits,
        b_stoch    = b_qspec.stoch,
        b_denorm   = b_qspec.denorm,
        b_freq     = (not reuse and b_qspec.freq),
        b_mode     = b_qspec.mode,
        b_bias_pad = b_qspec.bias_pad,
        b_stdv_mul = b_qspec.stdv_mul,
        b_logfile  = b_qspec.logfile,
        name       = name,
    )


@ops.RegisterGradient("Quantize")
def quantize_grad(op, dy):

    e = [get_entropy()] if op.get_attr("b_stoch") == 2 else []
    dx = quantize_op(dy, op.inputs[2], op.inputs[1], e,
        ebits      = op.get_attr("b_ebits"),
        fbits      = op.get_attr("b_fbits"),
        stoch      = op.get_attr("b_stoch"),
        denorm     = op.get_attr("b_denorm"),
        freq       = op.get_attr("b_freq"),
        mode       = op.get_attr("b_mode"),
        bias_pad   = op.get_attr("b_bias_pad"),
        stdv_mul   = op.get_attr("b_stdv_mul"),
        logfile    = op.get_attr("b_logfile"),
    )
    return (dx, None, None) if len(op.inputs) == 3 else (dx, None, None, None)


stat_headers = [
    "sat_pct",
    "ftz_pct",
    "max",
    "mean",
    "stdv",
    "mean+stdv5",
    "max_stat_lo",
    "max_stat_hi",
    "count",
    "name",
]

def log_stats(x, step, sat_val=65504.0, ftz_val=2.0**-24, freq=512, bfreq=512, logfile="", name=None):

    assert  freq == 0 or round(np.log2( freq)) == np.log2( freq)
    assert bfreq == 0 or round(np.log2(bfreq)) == np.log2(bfreq)

    # tack on timestamp if desired
    logfile = logfile % { "timestamp" : get_timestamp() }

    global log_init
    if logfile and logfile not in log_init:
        with open(logfile, 'w') as log:
            log.write("\t".join(stat_headers) + "\n")
        log_init.add(logfile)

    pow2 = int(np.log2(freq or bfreq))
    first_steps = [1 << p for p in range(pow2)]

    return log_stats_op(x, step,
        sat_val     = sat_val,
        ftz_val     = ftz_val,
        freq        = freq,
        bfreq       = bfreq,
        logfile     = logfile,
        first_steps = first_steps,
        name        = name or "log_stats")

@ops.RegisterGradient("LogStats")
def log_stats_grad(op, dy):

    dx = log_stats_op(dy, op.inputs[1],
        sat_val     = op.get_attr("sat_val"),
        ftz_val     = op.get_attr("ftz_val"),
        freq        = op.get_attr("bfreq"),
        bfreq       = op.get_attr("bfreq"),
        logfile     = op.get_attr("logfile"),
        first_steps = op.get_attr("first_steps"))
    return (dx, None)

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

