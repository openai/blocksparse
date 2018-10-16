
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
from blocksparse.quantize import quantize


data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))


############################## Adam #####################################

adam_op       = _op_module.adam
adam_gated_op = _op_module.adam_gated


def BlocksparseAdam(grads, params,
        lr=0.001, decay_mean=0.9, decay_var=0.999, epsilon=1e-8, grad_scale=1.0, clip_sigma=0.0,
        global_step=None, gated=False, param_qspec=None, mean_qspec=None, var_qspec=None):

    with tf.device("/cpu:0"), tf.variable_scope("adam_lr"):

        if global_step is None:
            t = tf.Variable(initial_value=0.0, name="t", trainable=False)
            t = t.assign_add(1.0)
        else:
            t = tf.cast(global_step.assign_add(1), tf.float32)

        lr = lr * tf.sqrt((1.0 - tf.pow(decay_var, t))) /  (1.0 - tf.pow(decay_mean, t))

        if type(grad_scale) is float:
            grad_scale = tf.constant(grad_scale)
        if type(clip_sigma) is float:
            clip_sigma = tf.constant(clip_sigma)

    updates = list()
    for grad, param in zip(grads, params):

        mean = slot_creator.create_zeros_slot(param, "adam_mean")
        var  = slot_creator.create_zeros_slot(param, "adam_variance")
        gate = getattr(param, "gate", None)

        colon = param.name.find(":")
        name  = param.name if colon < 0 else param.name[0:colon]

        with tf.device("/gpu:0"), tf.variable_scope("adam/" + name):
            if gated and gate is not None:
                op = adam_gated_op(gate, grad, param, mean, var, lr, grad_scale, clip_sigma,
                        decay_mean=decay_mean, decay_var=decay_var, epsilon=epsilon)
            else:
                op = adam_op(grad, param, mean, var, lr, grad_scale, clip_sigma,
                        decay_mean=decay_mean, decay_var=decay_var, epsilon=epsilon)

            if param_qspec is not None:
                updates.append(param.assign(quantize(op.out_param, param_qspec, name="param")))
            else:
                updates.append(op.out_param)

            if mean_qspec is not None:
                updates.append(mean.assign(quantize(op.out_mean, mean_qspec, name="mean")))

            if var_qspec is not None:
                updates.append(var.assign(quantize(op.out_var, var_qspec, name="var")))

    return tf.group(*updates)

def Adam(grads, params,
        lr=0.001, decay_mean=0.9, decay_var=0.999, epsilon=1e-8, grad_scale=1.0, clip_sigma=0.0,
        global_step=None, param_qspec=None, mean_qspec=None, var_qspec=None):

    return BlocksparseAdam(grads, params,
        lr=lr, decay_mean=decay_mean, decay_var=decay_var, epsilon=epsilon, grad_scale=grad_scale,
        global_step=global_step, param_qspec=param_qspec, mean_qspec=mean_qspec, var_qspec=var_qspec)

############################## Group LASSO / Blocksparse L2 decay #####################################

l2_decay_op       = _op_module.blocksparse_l2_decay
l2_decay_gated_op = _op_module.blocksparse_l2_decay_gated

class BlocksparseL2Decay(object):

    def __init__(self, rate=1e-5, gated=False, epsilon=1e-12):

        self.rate    = rate
        self.gated   = gated
        self.epsilon = epsilon

    def apply(self, grad_params, gpu=0):

        updates = []

        for grad, param in grad_params:

            # only apply to block-sparse tensors
            shape = param.get_shape().as_list()
            if len(shape) == 3 and shape[1] == shape[2] and shape[1] in (8,16,32):

                gate = getattr(param, "gate", None)

                with tf.device("/gpu:%d" % gpu), ops.name_scope("l2_decay"):
                    if self.gated and gate is not None:
                        op = l2_decay_gated_op(param, gate, self.rate, epsilon=self.epsilon)
                    else:
                        op = l2_decay_op(param, self.rate, epsilon=self.epsilon)

                    updates.append(op)

        return tf.group(*updates)


############################## Exponential Moving Average #####################################

ema_op       = _op_module.ema
ema_gated_op = _op_module.ema_gated

class BlocksparseEma(object):

    def __init__(self, decay=0.999, gated=False):
        self.decay    = decay
        self.gated    = gated
        self.averages = dict()

    def apply(self, grad_params, gpu=0, qspec=None):

        for grad, param in grad_params:
            with ops.init_scope():

                self.averages[param] = slot_creator.create_slot(param, param.initialized_value(), "ema")

                ops.add_to_collection(ops.GraphKeys.MOVING_AVERAGE_VARIABLES, param)

        ema_ops = []
        for grad, param in grad_params:

            colon = param.name.find(":")
            name  = param.name if colon < 0 else param.name[0:colon]

            with tf.device("/gpu:%d" % gpu), tf.variable_scope("ema/" + name):
                ema  = self.averages[param]
                gate = getattr(param, "gate", None)
                if self.gated and gate is not None:
                    op = ema_gated_op(ema, param, gate, decay=self.decay)
                else:
                    op = ema_op(ema, param, decay=self.decay)

                if qspec is not None:
                    ema_ops.append(ema.assign(quantize(op, qspec, name="ema")))
                else:
                    ema_ops.append(op)

        return tf.group(*ema_ops)

    def average(self, param):
        return self.averages.get(param, None)

############################## Blocksparse Pruning #####################################


maxnorm_prune_op = _op_module.blocksparse_maxnorm_prune

class BlocksparseMaxnormPrune(object):

    def __init__(self, threshold=1e-5, ema=None):

        self.threshold = threshold
        self.ema       = ema

    def apply(self, grad_params, gpu=0):

        updates = []
        for grad, param in grad_params:

            # only apply to gated block-sparse tensors
            gate = getattr(param, "gate", None)

            if gate is not None:

                # apply pruning to the moving average
                if self.ema is not None:
                    param = self.ema.average(param)

                with tf.device("/gpu:%d" % gpu), ops.name_scope("maxnorm_prune"):

                    updates.append(maxnorm_prune_op(gate, param, self.threshold))

        return tf.group(*updates)

############################## ClipAdamOptimizer #####################################


class ClipAdamOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_sigmas=0.0, grad_scale=1.0, sat_infs=None, zero_nans=True, name="ClipAdam"):
        super().__init__(False, name)
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self.sat_infs   = sat_infs
        self.zero_nans  = zero_nans
        self.name       = name

        with tf.device("/cpu:0"), tf.variable_scope("adam_beta"):

            if type(learning_rate) in (float, int):
                learning_rate = tf.constant(float(learning_rate))
            if type(clip_sigmas)   in (float, int):
                clip_sigmas   = tf.constant(float(clip_sigmas))
            if type(grad_scale)    in (float, int):
                grad_scale    = tf.constant(float(grad_scale))

            self.beta1_t     = tf.constant(beta1)
            self.beta2_t     = tf.constant(beta2)
            self.beta1_power = tf.Variable(initial_value=beta1, name="beta1_power", trainable=False)
            self.beta2_power = tf.Variable(initial_value=beta2, name="beta2_power", trainable=False)
            self.clip_sigma  = clip_sigmas
            self.grad_scale  = grad_scale
            self.lr          = learning_rate * tf.sqrt(1 - self.beta2_power) / (1 - self.beta1_power)

    def _get_beta_accumulators(self):
        return self.beta1_power, self.beta2_power

    def _non_slot_variables(self):
        return self._get_beta_accumulators()

    def _create_slots(self, params):
        # Create slots for the first and second moments.
        for param in params:
            self._zeros_slot(param, "m", self.name + "Mean")
            self._zeros_slot(param, "v", self.name + "Var")

    def _apply_dense(self, grad, param):

        m = self.get_slot(param, "m")
        v = self.get_slot(param, "v")

        # a float32 grad could still could contain infs from upstream fp16 math
        sat_infs = grad.dtype is tf.float16 if self.sat_infs is None else self.sat_infs

        return adam_op(grad, param, m, v, self.lr, self.grad_scale, self.clip_sigma,
            decay_mean=self.beta1, decay_var=self.beta2, epsilon=self.epsilon,
            sat_infs=sat_infs, zero_nans=self.zero_nans, lazy_update=hasattr(grad, "lazy")).out_param

    def _apply_sparse(self, grad, param):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies([ self.lr ]), tf.device("/cpu:0"):
            update_beta1 = self.beta1_power.assign(self.beta1_power * self.beta1_t)
            update_beta2 = self.beta2_power.assign(self.beta2_power * self.beta2_t)

        return tf.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AdamOptimizer(ClipAdamOptimizer):
    def __init__(self, learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8, grad_scale=1.0, sat_infs=None, zero_nans=True, name="Adam"):
        super().__init__(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, grad_scale=grad_scale, sat_infs=sat_infs, zero_nans=zero_nans, name=name)


############################## ClipAdamOptimizer #####################################

adafactor1d_op = _op_module.adafactor1d
adafactor2d_op = _op_module.adafactor2d

class AdafactorOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=5e-4, beta2=0.999, epsilon=1e-30, clip_thresh=1.0, grad_scale=1.0, sat_infs=None, zero_nans=True, name="Adafactor"):
        super().__init__(False, name)
        self.epsilon    = epsilon
        self.sat_infs   = sat_infs
        self.zero_nans  = zero_nans
        self.name       = name

        with tf.device("/cpu:0"), tf.variable_scope("adafactor_decay"):

            if type(learning_rate) in (float, int):
                learning_rate = tf.constant(float(learning_rate))
            if type(clip_thresh)   in (float, int):
                clip_thresh   = tf.constant(float(clip_thresh))
            if type(grad_scale)    in (float, int):
                grad_scale    = tf.constant(float(grad_scale))
            one = tf.constant(1.0)

            self.decay1_power = tf.Variable(initial_value=beta2,       name="decay1_power", trainable=False)
            self.decay2_power = tf.Variable(initial_value=beta2*beta2, name="decay2_power", trainable=False)
            self.learn_rate   = learning_rate
            self.clip_thresh  = clip_thresh
            self.grad_scale   = grad_scale
            self.decay_t      = tf.constant(beta2)
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

        # a float32 grad could still could contain infs from upstream fp16 math
        sat_infs = grad.dtype is tf.float16 if self.sat_infs is None else self.sat_infs

        if param.shape.ndims == 2 and param.shape[0].value > 1:

            cv = self.get_slot(param, "cv")
            rv = self.get_slot(param, "rv")

            return adafactor2d_op(param, cv, rv, grad,
                self.decay, self.learn_rate, self.grad_scale, self.clip_thresh,
                epsilon=self.epsilon, sat_infs=sat_infs, zero_nans=self.zero_nans).out_param

        elif param.shape.ndims == 1 or (param.shape.ndims == 2 and param.shape[0].value == 1):

            cv = self.get_slot(param, "cv")

            return adafactor1d_op(param, cv, grad,
                self.decay, self.learn_rate, self.grad_scale, self.clip_thresh,
                epsilon=self.epsilon, sat_infs=sat_infs, zero_nans=self.zero_nans).out_param
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


