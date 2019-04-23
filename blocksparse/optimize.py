
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training  import slot_creator
from tensorflow.python.training  import optimizer
from blocksparse.utils import _op_module, scalar_constant, is_param_casted
from blocksparse.ewops import float_cast
from blocksparse.quantize import quantize


############################## AdamOptimizer #####################################

adam_op = _op_module.adam
blocksparse_adam_op =_op_module.blocksparse_adam

class AdamOptimizer(optimizer.Optimizer):
    def __init__(self,
        learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_sigmas=0.0,
        norm_scale=None, grad_scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False,
        gated=False, param_qspec=None, mean_qspec=None, var_qspec=None,
        fp16=False, zero_init_variables=False, name="Adam"):

        super().__init__(False, name)
        self.beta1       = beta1
        self.beta2       = beta2
        self.epsilon     = epsilon
        self.saturate    = saturate
        self.zero_infs   = zero_infs
        self.zero_nans   = zero_nans
        self.gated       = gated
        self.param_qspec = param_qspec
        self.mean_qspec  = mean_qspec
        self.var_qspec   = var_qspec
        self.name        = name
        self.norm_scale  = [] if norm_scale is None else [norm_scale]
        self.fp16        = fp16

        beta1_init = 0.0 if zero_init_variables else beta1
        beta2_init = 0.0 if zero_init_variables else beta2

        with tf.device("/cpu:0"), tf.variable_scope("adam_beta"):

            one = scalar_constant(1.0, dtype=tf.float32)
            self.beta1_power = tf.Variable(initial_value=beta1_init, name="beta1_power", trainable=False)
            self.beta2_power = tf.Variable(initial_value=beta2_init, name="beta2_power", trainable=False)
            self.beta1_t     = scalar_constant(beta1,         dtype=tf.float32)
            self.beta2_t     = scalar_constant(beta2,         dtype=tf.float32)
            self.clip_sigma  = scalar_constant(clip_sigmas,   dtype=tf.float32)
            self.grad_scale  = scalar_constant(grad_scale,    dtype=tf.float32)
            self.lr          = scalar_constant(learning_rate, dtype=tf.float32) * tf.sqrt(one - self.beta2_power) / (one - self.beta1_power)

    def _get_beta_accumulators(self):
        return self.beta1_power, self.beta2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, params):
        # Create slots for the first and second moments.
        with tf.device("/gpu:0"), tf.control_dependencies(None):
            for param in params:
                # only use fp16 for larger params that benefit from memory savings
                dtype = tf.float16 if self.fp16 and param.shape.num_elements() >= 8*1024 else tf.float32  #is_param_casted(param)

                self._get_or_make_slot(param, tf.zeros(param.shape, dtype=dtype), "Mean", self.name)
                self._get_or_make_slot(param, tf.zeros(param.shape, dtype=dtype), "Var", self.name)

    def _apply_dense(self, grad, param):

        m = self.get_slot(param, "Mean")
        v = self.get_slot(param, "Var")

        gate = getattr(param, "gate", None)
        gate = [gate] if self.gated and gate is not None else []

        op = adam_op(grad, param, m, v, self.lr, self.grad_scale, self.clip_sigma, self.norm_scale, gate,
            decay_mean=self.beta1, decay_var=self.beta2, epsilon=self.epsilon,
            saturate=self.saturate, zero_infs=self.zero_infs, zero_nans=self.zero_nans, lazy_emb=hasattr(grad, "lazy"))

        updates = list()
        if self.param_qspec is not None:
            updates.append(param.assign(quantize(op.out_param, self.param_qspec, name="param_" + param.op.name)))
        else:
            updates.append(op.out_param)

        if self.mean_qspec is not None:
            updates.append(m.assign(quantize(op.out_mean, self.mean_qspec, name="mean_" + param.op.name)))

        if self.var_qspec is not None:
            updates.append(v.assign(quantize(op.out_var, self.var_qspec, name="var_" + param.op.name)))

        return tf.group(*updates) if len(updates) > 1 else updates[0]

    def _apply_sparse(self, grad, param):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies([ self.lr ]), tf.device("/cpu:0"):
            update_beta1 = self.beta1_power.assign(self.beta1_power * self.beta1_t)
            update_beta2 = self.beta2_power.assign(self.beta2_power * self.beta2_t)

        return tf.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


############################## AdafactorOptimizer #####################################

adafactor1d_op = _op_module.adafactor1d
adafactor2d_op = _op_module.adafactor2d

class AdafactorOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=5e-4, beta2=0.999, epsilon=1e-30, clip_thresh=1.0,
        norm_scale=None, grad_scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False,
        name="Adafactor", zero_init_variables=False):

        super().__init__(False, name)
        self.epsilon    = epsilon
        self.saturate   = saturate
        self.zero_infs  = zero_infs
        self.zero_nans  = zero_nans
        self.name       = name
        self.norm_scale = [] if norm_scale is None else [norm_scale]

        beta2_init = 0.0 if zero_init_variables else beta2

        with tf.device("/cpu:0"), tf.variable_scope("adafactor_decay"):

            one = scalar_constant(1.0, dtype=tf.float32)
            self.decay1_power = tf.Variable(initial_value=beta2_init,            name="decay1_power", trainable=False)
            self.decay2_power = tf.Variable(initial_value=beta2_init*beta2_init, name="decay2_power", trainable=False)
            self.learn_rate   = scalar_constant(learning_rate, dtype=tf.float32)
            self.clip_thresh  = scalar_constant(clip_thresh,   dtype=tf.float32)
            self.grad_scale   = scalar_constant(grad_scale,    dtype=tf.float32)
            self.decay_t      = scalar_constant(beta2,         dtype=tf.float32)
            self.decay        = self.decay_t * (one - self.decay1_power) / (one - self.decay2_power)

    def _get_beta_accumulators(self):
        return self.decay1_power, self.decay2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, params):
        # Create slots for the first and second moments.
        for param in params:
            if param.shape.ndims == 2 and param.shape[0].value > 1:
                self._get_or_make_slot(param, tf.zeros(param.shape[1].value), "cv", self.name + "CV")
                self._get_or_make_slot(param, tf.zeros(param.shape[0].value), "rv", self.name + "RV")
            elif param.shape.ndims == 1 or (param.shape.ndims == 2 and param.shape[0].value == 1):
                self._get_or_make_slot(param, tf.zeros(param.shape.num_elements()), "cv", self.name + "CV")
            else:
                raise ValueError("only 1 or 2d params are supported")

    def _apply_dense(self, grad, param):

        if param.shape.ndims == 2 and param.shape[0].value > 1:

            cv = self.get_slot(param, "cv")
            rv = self.get_slot(param, "rv")

            return adafactor2d_op(param, cv, rv, grad,
                self.decay, self.learn_rate, self.grad_scale, self.clip_thresh, self.norm_scale, epsilon=self.epsilon,
                saturate=self.saturate, zero_infs=self.zero_infs, zero_nans=self.zero_nans).out_param

        elif param.shape.ndims == 1 or (param.shape.ndims == 2 and param.shape[0].value == 1):

            cv = self.get_slot(param, "cv")

            return adafactor1d_op(param, cv, grad,
                self.decay, self.learn_rate, self.grad_scale, self.clip_thresh, self.norm_scale, epsilon=self.epsilon,
                saturate=self.saturate, zero_infs=self.zero_infs, zero_nans=self.zero_nans).out_param
        else:
            raise ValueError("only 1 or 2d params are supported")

    def _apply_sparse(self, grad, param):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies([ self.decay ]), tf.device("/cpu:0"):
            update_decay1 = self.decay1_power.assign(self.decay1_power * self.decay_t)
            update_decay2 = self.decay2_power.assign(self.decay2_power * self.decay_t)

        return tf.group(*update_ops + [update_decay1, update_decay2], name=name_scope)

############################## ClipGlobalNorm #####################################

clip_global_norm_op = _op_module.clip_global_norm

def clip_by_global_norm(grads, clip_norm=1.0, grad_scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False):

    grad_float = list()
    grad_ehalf = list()
    grad_bhalf = list()

    for grad in grads:
        if   grad.dtype is tf.float32:
            grad_float.append(grad)
        elif grad.dtype is tf.float16:
            grad_ehalf.append(grad)
        elif grad.dtype is tf.bfloat16:
            grad_bhalf.append(grad)
        else:
            raise ValueError("unsupported grad dtype")

    with tf.device("/gpu:0"):
        global_norm, norm_scale, _ = clip_global_norm_op(
            scalar_constant(grad_scale, dtype=tf.float32),
            scalar_constant(clip_norm,  dtype=tf.float32),
            grad_float, grad_ehalf, grad_bhalf,
            saturate=saturate, zero_infs=zero_infs, zero_nans=zero_nans)

    return global_norm, norm_scale

def global_norm(grads, grad_scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False):
    gn, _ = clip_by_global_norm(grads, clip_norm=9e9, grad_scale=grad_scale, saturate=saturate, zero_infs=zero_infs, zero_nans=zero_nans)
    return gn

# old function name
def ClipGlobalNorm(grads, clip_norm=1.0, grad_scale=1.0, saturate=0.0, zero_infs=False, zero_nans=False):
    return clip_by_global_norm(grads, clip_norm=clip_norm, grad_scale=grad_scale, saturate=saturate, zero_infs=zero_infs, zero_nans=zero_nans)


############################## Exponential Moving Average #####################################

ema_op = _op_module.ema

class Ema(object):

    def __init__(self, decay=0.999, gated=False, fp16=False, name="Ema"):
        self.decay    = decay
        self.gated    = gated
        self.fp16     = fp16
        self.name     = name
        self.averages = dict()

    def apply(self, params, qspec=None):

        with tf.device("/gpu:0"), tf.control_dependencies(None):
            for param in params:
                if self.fp16 == 2 or (self.fp16 and is_param_casted(param)):
                    # only use fp16 for params that are explicitly cast to fp16 before use
                    init  = float_cast(param.initialized_value(), dtype=tf.float16)
                    dtype = tf.float16
                else:
                    init  = param.initialized_value()
                    dtype = tf.float32

                with tf.variable_scope(None, param.op.name + "/" + self.name):
                    # use the Identity read op output as the key
                    # this lets us lookup ema vars by Cast op outputs
                    self.averages[param.value()] = tf.get_variable("ema", dtype=dtype, initializer=init, trainable=False)
                ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, param)

        ema_ops = []
        for param in params:

            ema   = self.averages[param.value()]
            gate  = getattr(param, "gate", None)
            gate  = [gate] if self.gated and gate is not None else []

            op = ema_op(ema, param, gate, decay=self.decay)

            if qspec is not None:
                ema_ops.append(ema.assign(quantize(op, qspec, name="ema_" + param.op.name)))
            else:
                ema_ops.append(op)

        return tf.group(*ema_ops)

    def average(self, param):
        if isinstance(param, tf.Variable):
            # this is just a raw param
            key = param.value()
        elif isinstance(param, tf.Tensor):
            # we're given a Cast op output
            # TODO: maybe traverse deeper?
            key = param.op.inputs[0]
        else:
            raise TypeError("bad param type")

        return self.averages.get(key, None)


############################## Group LASSO / Blocksparse L2 decay #####################################

l2_decay_op = _op_module.blocksparse_l2_decay

def _check_param_shape(param, gate=None):
    assert len(param.shape) == 3 and param.shape[1].value == param.shape[2].value and param.shape[1].value in (8,16,32,64)
    if gate is not None:
        assert gate.shape.num_elements() == param.shape[0].value

def blocksparse_l2_decay(param, gate=None, rate=0.05, epsilon=1e-12):

    _check_param_shape(param, gate)

    gate = [gate] if gate is not None else []

    return l2_decay_op(param, scalar_constant(rate, dtype=tf.float32), gate, epsilon=epsilon)

############################## Blocksparse Pruning #####################################

blocksparse_norm_op            = _op_module.blocksparse_norm
blocksparse_prune_op           = _op_module.blocksparse_prune
blocksparse_threshold_prune_op = _op_module.blocksparse_threshold_prune

def blocksparse_norm(param, norm="max"):
    _check_param_shape(param)
    return blocksparse_norm_op(param, norm_type=1 if norm.lower() == "l2" else 0)

def blocksparse_prune(param, gate, step, sparsity=None, threshold=None, norm="max", frequency=1):

    _check_param_shape(param, gate)

    # one must be set
    assert (sparsity is None) ^ (threshold is None)

    if sparsity is not None:

        # apply pruning to the moving average
        norms = blocksparse_norm(param, norm=norm)

        k = scalar_constant(param.shape[0].value, dtype=tf.int32)

        _, idx = tf.nn.top_k(norms, k=k, sorted=True)

        return blocksparse_prune_op(gate, idx, scalar_constant(sparsity, dtype=tf.float32), step, frequency=frequency)

    elif threshold is not None:

        norm = 1 if norm.lower() == "l2" else 0

        return blocksparse_threshold_prune_op(gate, param, scalar_constant(threshold, dtype=tf.float32), step, frequency=frequency, norm_type=norm)
