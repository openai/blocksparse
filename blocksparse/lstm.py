
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from blocksparse.utils import _op_module


############################## fused_lstm_gates #####################################

lstm_gates_op       = _op_module.lstm_gates
lstm_gates_grad_op  = _op_module.lstm_gates_grad
lstm_gates4_op      = _op_module.lstm_gates4
lstm_gates4_grad_op = _op_module.lstm_gates4_grad
bias_grad_op        = _op_module.bias_grad

def fused_lstm_gates(c, *args, bias=None, forget_bias=1.0, name=None):
    # returns c_next, h_next

    dev = args[0].op.device.lower()
    if not dev or "cpu" in dev:
        h = args[0]
        if bias is not None:
            h = tf.nn.bias_add(h, bias)

        i, j, f, o = tf.split(h, 4, axis=1)

        fb = tf.constant(forget_bias, dtype=f.dtype)

        new_c = tf.add(tf.multiply(c, tf.sigmoid(tf.add(f, fb))), tf.multiply(tf.sigmoid(i), tf.tanh(j)))
        new_h = tf.multiply(tf.tanh(new_c), tf.sigmoid(o))
        return new_c, new_h

    # args is h (all four gates fused in single tensor)
    if len(args) == 1:
        bias = [] if bias is None else [ bias ]
        return lstm_gates_op(c, args[0], bias, forget_bias=forget_bias, name=name)

    assert len(args) == 4, "args are i, u, f, o"
    assert bias is None, "bias not enabled in this mode"
    return lstm_gates4_op(c, *args, forget_bias=forget_bias, name=name)


@ops.RegisterGradient("LSTMGates")
def fused_lstm_gates_grad(op, ec, eh):

    bias = [] if len(op.inputs) == 2 else [ op.inputs[2] ]

    # in our kernels we just conditionaly load zero instead of reading the constant tensor
    grads = [eh] if ec is None or ec.op == "Fill" else [eh, ec]

    dc, dh = lstm_gates_grad_op(op.inputs[0], op.inputs[1], bias, grads, forget_bias=op.get_attr("forget_bias") )

    if len(op.inputs) == 2:
        return dc, dh

    # compute bias grad
    #db = ew_db_dzb_op(dh, op.inputs[2], op=BIASADD_OP)
    # db = bias_grad_op(dh, op.inputs[2])
    db, _ = bias_grad_op(dh, op.inputs[2], axis=1)

    return dc, dh, db


@ops.RegisterGradient("LSTMGates4")
def fused_lstm_gates4_grad(op, ec, eh):
    # in our kernels we just conditionaly load zero instead of reading the constant tensor
    grads = [eh] if ec is None or ec.op == "Fill" else [eh, ec]
    return lstm_gates4_grad_op(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4], grads, forget_bias=op.get_attr("forget_bias") )


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


############################## Sparse Relu #####################################


sparse_relu_op = _op_module.sparse_relu
ew_dx_dzza_op  = _op_module.ew_dx_dzza

def sparse_relu(x, alpha=1.0):
    return sparse_relu_op(x, alpha)

@ops.RegisterGradient("SparseRelu")
def sparse_relu_grad(op, dz):
    # same grad as relu
    return ew_dx_dzza_op(dz, op.outputs[0], op=RELU_OP)

def sparse_relu_test(x, alpha=1.0):

    axis = len(x.shape)-1
    mean = np.mean(x, axis=axis, keepdims=True)
    std  = np.std(x, axis=axis, keepdims=True)
    cutoff = mean + alpha*std
    return np.maximum(np.maximum(x, cutoff) - cutoff, 0.0)


############################## Fused BasicLSTMCell #####################################

from tensorflow.python.ops.rnn_cell import BasicLSTMCell, LSTMStateTuple

class FusedBasicLSTMCell(BasicLSTMCell):

  def __init__(self, *args, **kwargs):
    super(FusedBasicLSTMCell, self).__init__(*args, **kwargs)

  def call(self, inputs, state):

    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = tf.split(value=state, num_or_size_splits=2, axis=one)

    h = tf.matmul( tf.concat([inputs, h], 1), self._kernel )

    c, h = fused_lstm_gates(c, h, bias=self._bias, forget_bias=self._forget_bias)

    if self._state_is_tuple:
      state = LSTMStateTuple(c, h)
    else:
      state = tf.concat([c, h], 1)

    return h, state


############################## Simple Fused LSTM with optional layernorm #####################################



def grouped_lstm(inputs, width, timesteps, initial_state, scope="grouped_lstm", reuse=None, lstm_id=0, layernorm=True):

    fp16 = inputs.dtype is tf.float16

    if layernorm:
        from blocksparse.norms import layer_norm
    if fp16:
        from blocksparse.ewops import float_cast

    in_width = inputs.shape[-1].value

    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable('kernel', shape=[in_width + width, 4 * width])
        b = tf.get_variable('bias', shape=[4 * width])
        if layernorm:
            g = tf.get_variable('gain', shape=[4 * width])

        c, h = initial_state

        if fp16:
            w = float_cast(w, dtype=tf.float16, dx_dtype=tf.float16)

        if timesteps > 1:
            inputs = [ tf.squeeze(x) for x in tf.split(inputs, timesteps, axis=1) ]
        else:
            inputs = [ tf.reshape(inputs, [-1, inputs.shape[-1].value]) ]

        outputs = []
        for t, x in enumerate(inputs):

            h = tf.matmul( tf.concat([x, h], 1), w, name="lstm_%02d/step_%04d" % (lstm_id, t))

            if layernorm:
                h    = layer_norm(h, g, b, axis=1, segments=4)
                c, h = fused_lstm_gates(c, h, forget_bias=1.0)
            else:
                c, h = fused_lstm_gates(c, h, bias=b, forget_bias=1.0)

            outputs.append(h)

        output = tf.stack(outputs, axis=1)

    return output, [c, h]


def group_lstm_grads(grads, params, scope="grouped_lstm", group_size=None):

    grad = None
    grad_idx = None
    for i, (g, p) in enumerate(zip(grads, params)):
        if scope in p.name and "kernel" in p.name:
            grad = g
            grad_idx = i
            break
    assert grad is not None

    # backward walk param grad to find dw MatMul ops
    # walk should terminate with each MatMul op
    ops  = list()
    wave = set([grad.op])
    while wave:
        new_wave = set()
        for op in wave:
            for op in (t.op for t in op.inputs):
                # TN MatMul ops
                if op.type == "MatMul" and op.get_attr("transpose_a") and not op.get_attr("transpose_b"):
                    ops.append(op)
                else:
                    new_wave.add(op)
        wave = new_wave

    # sort op names descending and split out the lstms (if weights are shared)
    last_lstm = None
    lstms = list()
    ops.sort(key=lambda op: op.name, reverse=True)
    for op in ops:
        # gradients/grouped_lstm/lstm_2/step_00_grad/MatMul_1 => lstm_2
        lstm = op.name.split("/")[-3]
        if last_lstm != lstm:
            lstms.insert(0, list())
            last_lstm = lstm
        lstms[0].append(op)

    # we're going to be using absolute names, so clear name_scope
    with tf.name_scope(None):

        lstm_grads = list()
        for lstm_ops in lstms:

            # default dw op to one big matmul per lstm
            if group_size is None:
                group_size = len(lstm_ops)

            # use the lstm scope for the new ops
            # gradients/grouped_lstm/lstm_2/step_00_grad/MatMul_1 => gradients/grouped_lstm/lstm_2
            scope = lstm_ops[-1].name.split('/')
            scope = '/'.join(scope[0:-2])

            offset = 0
            while offset < len(lstm_ops):

                xs = tf.concat([op.inputs[0] for op in lstm_ops[offset:offset+group_size] ], axis=0)
                gs = tf.concat([op.inputs[1] for op in lstm_ops[offset:offset+group_size] ], axis=0)

                mmop = tf.matmul(xs, gs, transpose_a=True, transpose_b=False, name="%s/dw_%04d" % (scope, offset))
                grad = mmop if offset == 0 else ew.add(grad, mmop, name="%s/add_%04d" % (scope, offset))

                offset += group_size

            lstm_grads.append(grad)

        if len(lstms) > 1:
            from blocksparse.ewops import add_n
            # gradients/grouped_lstm/lstm_2/step_00_grad/MatMul_1 => gradients/grouped_lstm
            scope = lstms[0][-1].name.split('/')
            scope = '/'.join(scope[0:-3])
            grads[grad_idx] = tf.add_n(lstm_grads, name="%s/add_n" % scope)
        else:
            grads[grad_idx] = lstm_grads[0]

    #grads modified in place





# lstm_scopes = dict()
    # # rediculous amount of code just to be able to re-enter a variable scope without its name being re-numbered.
    # # https://github.com/tensorflow/tensorflow/pull/14390
    # global lstm_scopes
    # if scope not in lstm_scopes:
    #     with tf.variable_scope(scope) as lstm_scope:
    #         lstm_scopes[scope] = lstm_scope
    # lstm_scope = lstm_scopes[scope]

    # with tf.variable_scope(lstm_scope, auxiliary_name_scope=False), tf.name_scope(lstm_scope.original_name_scope):
    #     with tf.variable_scope(weights_scope, reuse=weights_reuse):
    #         w = tf.get_variable('kernel', shape=[in_width + width, 4 * width])
    #         if bias_scope is None:
    #             b = tf.get_variable('bias', shape=[4 * width])
    #             if layernorm:
    #                 g = tf.get_variable('gain', shape=[4 * width])

    #     if bias_scope is not None:
    #         with tf.variable_scope(bias_scope, reuse=bias_reuse):
    #             b = tf.get_variable('bias', shape=[4 * width])
    #             if layernorm:
    #                 g = tf.get_variable('gain', shape=[4 * width])