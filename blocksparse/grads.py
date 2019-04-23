
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import collections
import tensorflow  as tf
from tensorflow.python.framework import ops

from blocksparse.utils import _op_module
import blocksparse.ewops as ew

recompute_op = _op_module.recompute

# Recompute Decorator
class recomputable(object):

    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func

    def __call__(self, *args, **kwargs):

        # toggle recompute on and off with the recompute keyword arg
        recompute = kwargs.pop("recompute", False)

        # generate the forward pass portion of graph
        fwd = self.func(*args, **kwargs)
        if not recompute:
            return fwd

        # create a temp op to be a control input to the recomputed graph
        with tf.device("/cpu:0"):
            ctrl_op = tf.constant(0.0, name="temp_ctrl_op").op

        # Enable variable reuse in the current variable_scope.
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            # Distinguish these ops in a new name_scope
            with tf.name_scope("recompute"):
                # Use the temp ctrl_op to track needed control input targets
                with tf.control_dependencies([ctrl_op]):
                    # Generate the recomputed ops that we want to run in the backward pass.
                    bwd = self.func(*args, **kwargs)

        # the recompute op is a passthrough op for the fwd inputs.
        # the bwd inputs allow our custom grad fuction to redirect gradient flow over these
        # bwd inputs are disconnected after gradients are generated
        y = recompute_op(_AsList(fwd), _AsList(bwd), name=self.func.__name__)

        # hold on to the temp for setting up control dependencies in the grad op.
        y[0].op.ctrl_op = ctrl_op

        return y[0] if len(y) == 1 else y

    def __get__(self, instance, owner):
        # Necessary for the decorator to work on instance methods.
        # See https://stackoverflow.com/questions/30104047/how-can-i-decorate-an-instance-method-with-a-decorator-class
        return functools.partial(self.__call__, instance)

@ops.RegisterGradient("Recompute")
def recompute_grad(op, *dys):

    # Ensure recompute portion of graph is only executed in the backward pass just prior to use.
    dy_ops = [dy.op for dy in dys]

    # our temp ctrl_op points to exactly the ops that need to be executed after dys ops
    for recompute_op in op.ctrl_op._control_outputs:

        # rebild control_inputs list for this op filering out the temp ctrl_op
        ctrl_inputs = [x for x in recompute_op.control_inputs if x != op.ctrl_op]

        # rebuild control_inputs from scratch
        recompute_op._remove_all_control_inputs()

        # no need to hold up simple scalar/vector constants
        if recompute_op.type == "Const" and len(recompute_op.outputs[0].shape) < 2:
            if len(ctrl_inputs):
                recompute_op._add_control_inputs(ctrl_inputs)
        else:
            # tack on dy ops
            recompute_op._add_control_inputs(ctrl_inputs + dy_ops)

    # done with temp ctrl_op
    op.ctrl_op = None

    # direct the gradient flow over the recomputed ops (skipping the forward graph)
    return [None]*len(op.outputs) + list(dys)


def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]

def _SetGrad(grads, x, dx):
    op = x.op
    op_grads = grads.get(op)
    if op_grads is None:
        # for each op output, maintain a list of gradient inputs
        grads[op] = op_grads = [[] for _ in op.outputs]
    # add this grad to the appropriate list
    op_grads[x.value_index].append(dx)

def _GetGrad(grads, x):
    op_grads = grads.get(x.op)
    if op_grads is None:
        return None
    # this should always return the _AggregatedGrads value instead of the list
    return op_grads[x.value_index]

def _AggregatedGrads(grads, op, agg_size):

    # convert lists of gradient inputs to tensors
    dys = grads.get(op)
    if dys is not None:
        for i, dy in enumerate(dys):
            if len(dy):
                if len(dy) == 1:
                    dys[i] = dy[0]  # no op
                else:
                    # tf.add_n has poor accuracy in fp16
                    # also, we want to group accumuations so we can start freeing up memory early
                    # dys[i] = tf.add_n(dy)
                    agg = ew.add_n8_op(dy[0:agg_size]) if agg_size > 1 else dy[0]
                    for j in range(agg_size, len(dy), agg_size):
                        agg = ew.add_n8_op(dy[j:j+agg_size] + [agg])
                    dys[i] = agg
            else:
                dys[i] = None
        return dys
    else:
        return [None] * len(op.outputs)

def _PendingCount(ys_ops, xs_ops):

    grad_dtypes = set((tf.float32, tf.float16, tf.bfloat16))

    # Ascend tree from the params and/or inputs (xs) to the losses (ys).
    # Create set of each unique node along the way.
    reached_ops = set()
    queue = collections.deque(xs_ops)
    while queue:
        op = queue.popleft()
        if op not in reached_ops:
            reached_ops.add(op)
            for output in op.outputs:
                if output.dtype.base_dtype in grad_dtypes:
                    queue.extend(output.consumers())
    # Get the subset of ys are reachable from xs.
    reachable_ys_ops = set(op for op in ys_ops if op in reached_ops)

    # Descend tree from ys along the reachable path.
    # Mark unique ops along the way (between_ops).
    # Handle gradient rerouting for recompute nodes.
    recompute_ops = list()
    between_ops   = set()
    queue         = collections.deque(reachable_ys_ops)
    while queue:
        op = queue.popleft()
        if op in reached_ops:
            between_ops.add(op)
            # don't add the inputs again.
            reached_ops.remove(op)
            # For recompute ops only traverse the second graph copy
            # We don't want the forward pass ops contributing to the pending_count.
            if op.type == "Recompute":
                recompute_ops.append(op)
                n_outs = len(op.outputs)
                for x in op.inputs[n_outs:n_outs*2]:
                    queue.append(x.op)
            else:
                for x in op.inputs:
                    queue.append(x.op)

    # Build a mapping from operation to the number of grad inputs to that op
    # ops not in this dict should no longer be traversed (excepting the initial ys ops with no dependancies).
    pending_count = dict()
    for op in between_ops:
        for x in op.inputs:
            if x.op in between_ops:
                pending_count[x.op] = pending_count.get(x.op, 0) + 1

    return pending_count, reachable_ys_ops, recompute_ops

def _MatMulGradNN(op, dy):
    # Custom Gradient for MatMul (NN)
    # Force param gradient first so all-reduce can happen quicker.
    x = op.inputs[0]
    w = op.inputs[1]

    dw = tf.matmul(x, dy, transpose_a=True)
    with tf.control_dependencies([dw.op]):
        dx = tf.matmul(dy, w, transpose_b=True)

    return dx, dw

def gradients(ys, xs, grad_ys=None, stop_grads=None, group_aggregations=8, custom_matmul_grad=True):

    if group_aggregations > 8 or group_aggregations < 1:
        raise ValueError("gradients: group_aggregation sizes of 1-8 supported.")

    ys = _AsList(ys)
    xs = [x.value() if isinstance(x, tf.Variable) else x for x in _AsList(xs)]

    stop_grads = [] if stop_grads is None else _AsList(stop_grads)

    grad_ys = [None] * len(ys) if grad_ys is None else _AsList(grad_ys)
    assert len(ys) == len(grad_ys)

    with ops.name_scope("gradients"):

        for i, dy in enumerate(grad_ys):
            if dy is None:
                # float grads start at ones by default
                grad_ys[i] = tf.fill(tf.shape(ys[i]), tf.constant(1.0, dtype=ys[i].dtype, name=f"grad_ys_{i}"))

        ys_ops = [t.op for t in ys]
        xs_ops = [t.op for t in xs]

        pending_count, reachable_ys_ops, recompute_ops = _PendingCount(ys_ops, xs_ops)

        # The set of ops that terminate the gradient computation.
        # Confirm that our xs tensors are just endpoints in the graph.
        # Also set any externally provided stop grad ops.
        stop_ops = set(t.op for t in stop_grads)
        for op in xs_ops:
            is_stop_op = True
            for x in op.inputs:
                if x.op in pending_count:
                    is_stop_op = False
                    break
            if is_stop_op:
                stop_ops.add(op)

        # Each op output has an associated list of gradient inputs
        # If more than one, these need to be accumulated.
        # Add the initial gradients for the ys.
        grads = dict()
        for y, dy in zip(ys, grad_ys):
            _SetGrad(grads, y, dy)

        # Add the unique ys ops that are ready into the queue.
        queue = collections.deque()
        for op in reachable_ys_ops:
            # an op is ready if it has no dependecies
            if op not in pending_count:
                queue.append(op)

        while queue:
            op = queue.popleft()

            # only pending_count==0 ops are in the queue so all grad input lists are fully populated
            # go ahead and apply any needed add_n ops to these lists.
            dys = _AggregatedGrads(grads, op, group_aggregations)

            # confirm that we have at least one tensor to compute and that this isn't a stop grad op
            if any(dy is not None for dy in dys) and op not in stop_ops:
                # get the grad function for this op
                try:
                    if custom_matmul_grad and op.type == "MatMul" and not op.get_attr("transpose_a") and not op.get_attr("transpose_b"):
                        grad_fn = _MatMulGradNN
                    else:
                        grad_fn = ops.get_gradient_function(op)
                except LookupError:
                    raise LookupError(f"No gradient defined for operation '{op.name}' (op type: {op.type})")

                # for any missing input grads, build a zero input of the right dtype/shape
                for i, dy in enumerate(dys):
                    if dy is None:
                         dys[i] = tf.zeros_like(op.outputs[i])

                # call the grad function with the forward op node and list of grad inputs
                with ops.name_scope(op.name + "_grad"):
                    dxs = _AsList(grad_fn(op, *dys))

                    if len(dxs) != len(op.inputs):
                        raise ValueError(f"Num gradients {len(dxs)} generated for op {op.node_def} do not match num inputs {len(op.inputs)}")

                    #_LogOpGradients(op, dys, dxs)
            else:
                dxs = [None] * len(op.inputs)

            for i, (x, dx) in enumerate(zip(op.inputs, dxs)):
                if dx is not None:
                    # force unsorted_segment_sum call
                    if isinstance(dx, ops.IndexedSlices):
                        dx = tf.convert_to_tensor(dx)
                        #dx = emb.embedding_lookup_grad_op(dx.values, dx.indices, dx.dense_shape[0])

                    # do some shape sanity checking
                    try:
                        dx.set_shape(x.shape)
                    except ValueError:
                        raise ValueError("Incompatible shapes between op input {x.shape} and calculated input gradient {dx.shape} for {op.name} (idx:{i})")

                    # update the input grad list for the consumer of this gradient
                    _SetGrad(grads, x, dx)

            # Update pending count for the inputs of op and enqueue any ready ops
            for x in op.inputs:
                # only traverse nodes that are in the reachable gradient path (and hence have a pending entry)
                count = pending_count.get(x.op)
                if count is not None:
                    if count == 1:
                        # when count is 1 this should be last time we reach this node
                        queue.append(x.op)
                    pending_count[x.op] = count - 1

    # Disconnect the recomputed portion of the graph from the forward pass.
    # This was only needed to direct the gradient flow.
    # Leaving these connections in place would create a circular dependancy (from added control inputs).
    for op in recompute_ops:
        # Just overwrite the backward inputs with a copy of the forward inputs.
        n_out = len(op.outputs)
        for i, x in enumerate(op.inputs[:n_out]):
            op._update_input(i+n_out, x)

    return [_GetGrad(grads, x) for x in xs]


