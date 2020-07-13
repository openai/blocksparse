"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from mpi4py import MPI
from tensorflow.python.framework import ops
from blocksparse.utils import _op_module
from blocksparse.ewops import float_cast
from collections import deque

############################## Nccl Ops #####################################

op_counter = 0
init_num_comms = None
init_prereduce = None


# FIXME(taehoon): remove this
def mpi_size():
    return MPI.COMM_WORLD.Get_size()


def allreduce(x, sync_size=0, num_comms=2, logfile="", rank=0, prereduce=0, name=None,
              mpi_ranks=None, comm_id=0, debug_str=''):
    if mpi_ranks is None:
        mpi_ranks = list(range(0, mpi_size()))

    assert not x.device or x.device[
                              -2:] == ":0", "Only one gpu per process currently supported by allreduce: " + x.device
    global op_counter
    global init_num_comms
    global init_prereduce

    if init_num_comms is None:
        init_num_comms = num_comms
    elif init_num_comms != num_comms:
        print(
            "Warning: only the first value of num_comms (%d) that was passed in will be used.  num_comms=%d vale ignored." % (
            init_num_comms, num_comms))

    if init_prereduce is None:
        init_prereduce = prereduce
    elif init_prereduce != prereduce:
        print(
            "Warning: only the first value of prereduce (%d) that was passed in will be used.  prereduce=%d vale ignored." % (
            init_prereduce, prereduce))

    if logfile and rank == 0:
        print("%03d %s" % (op_counter, x.name))
    ret = _op_module.allreduce_nccl(x, op_num=op_counter, sync_size=sync_size, num_comms=num_comms, prereduce=prereduce, logfile=logfile, name=name, mpi_ranks=mpi_ranks, comm_id=comm_id, debug_str=debug_str)
    op_counter += 1
    return ret

# @ops.RegisterGradient("AllreduceNccl")
# def allreduce_grad(op, dy):
#     global op_counter

#     dx = _op_module.allreduce_nccl(dy,
#         op_num      = op_counter,
#         sync_size   = op.get_attr('sync_size'),
#         num_comms   = op.get_attr('num_comms'),
#         prereduce   = op.get_attr('prereduce'),
#         logfile     = op.get_attr('logfile'))

@ops.RegisterGradient("AllreduceNccl")
def allreduce_grad(op, dy):
    global op_counter
    global init_num_comms
    global init_prereduce

    sync_size = op.get_attr('sync_size')
    num_comms = op.get_attr('num_comms')
    prereduce = op.get_attr('prereduce')
    logfile = op.get_attr('logfile')
    mpi_ranks = op.get_attr('mpi_ranks')
    mpi_rank = op.get_attr('mpi_rank')
    comm_id = op.get_attr('comm_id')
    debug_str = op.get_attr('debug_str')

    dx = _op_module.allreduce_nccl(
        dy, op_num=op_counter,
        sync_size=sync_size,
        num_comms=num_comms,
        prereduce=prereduce,
        logfile=logfile,
        mpi_ranks=mpi_ranks,
        mpi_rank=mpi_rank,
        comm_id=comm_id,
        debug_str=debug_str,
    )

    op_counter += 1
    return dx


def group_allreduce(
        grads, parms, search_strings=None, cast_map=None, cast_all=None, allreduce_op=allreduce, **allreduce_kwargs):
    # if no grouping specified, create one group to reduce at the end (no overlap with compute)
    if search_strings is None:
        search_strings = ["group_allreduce_all"]

    groups = [(names, list(), list()) for names in search_strings]

    last_group_idx = len(groups) - 1

    for i, (grad, param) in enumerate(zip(grads, parms)):
        for j, (names, group16, group32) in enumerate(groups):
            # each group can be a single string, or a list of strings
            # TODO: support regex's
            if isinstance(names, str):
                names = (names,)

            if j == last_group_idx or any(name in param.name for name in names):

                if cast_all is not None:
                    grad = float_cast(grad, dtype=cast_all)

                elif cast_map is not None and name in cast_map:
                    grad = float_cast(grad, dtype=cast_map[name])

                if grad.dtype.base_dtype is tf.float16:
                    group16.append((i, grad, param))
                else:
                    group32.append((i, grad, param))
                break

    for name, group16, group32 in groups:
        count = 0
        if isinstance(name, str):
            str_name = name
        else:
            str_name = "_".join(name)
        str_name = str_name.replace('/', '_')
        for group in (group16, group32):
            count += len(group)
            if len(group) > 0:
                if len(group) == 1:
                    concated = group[0][1]
                else:
                    concated = tf.concat([tf.reshape(grad, [-1]) for _, grad, _ in group], 0, name="concat_"+str_name)

                reduced = allreduce_op(concated, **allreduce_kwargs)

                if len(group) == 1:
                    grads[group[0][0]] = reduced
                else:
                    offset = 0
                    for i, grad, param in group:
                        size = param.shape.num_elements()
                        grads[i] = tf.reshape(reduced[offset: offset + size], param.shape)
                        offset += size

        if count == 0:
            print("Warning: no grads found for all_reduce group: ", name)

    # grads modified in place, but return anyway
    return grads


def sync_variables_op(mpi_rank, num_comms=2, prereduce=0):
    ops = list()
    prev = []
    with tf.device("/gpu:0"):
        for var in tf.trainable_variables():
            with tf.control_dependencies(prev):
                op = tf.assign(var,
                               allreduce(var if mpi_rank == 0 else var * 0.0, num_comms=num_comms,
                                         prereduce=prereduce))
            prev = [op]
            ops.append(op)

    return tf.group(*ops)

def sync_globals_zero_init_op(num_comms=2, prereduce=0):
    ops  = list()
    prev = []
    with tf.device("/gpu:0"):
        for var in tf.global_variables():
            if var.dtype.base_dtype not in [tf.float32, tf.float16]:
                cast_back = True
                to_reduce = tf.cast(var, tf.float32)
            else:
                to_reduce = var
                cast_back = False
            with tf.control_dependencies(prev):
                reduced = allreduce(to_reduce, num_comms=num_comms, prereduce=prereduce)
            if cast_back:
                reduced = tf.cast(reduced, var.dtype.base_dtype)
            op = tf.assign(var, reduced)
            prev = [op]
            ops.append(op)

    return tf.group(*ops)

# These ops are always at the end of the input (parent) chains.
# We don't partiuarly care about their ordering.
def _skip_op(op):
    if op.type in ("Const", "VariableV2"):
        return True
    if op.type == "Identity" and op.name[-4:] == "read":
        return True
    return False

# The input and control_input ops are the parents of this op (output as a set)
def _get_parents_set(op):
    parents = set(i.op for i in op.inputs         if not _skip_op(i.op))
    parents.update(ci for ci in op.control_inputs if not _skip_op(ci  ))
    return parents

# The input and control_input ops are the parents of this op (output as a list)
def _get_parents_list(op):
    parents = [i.op for i in op.inputs if not _skip_op(i.op)]
    for ci in op.control_inputs:
        if not _skip_op(ci):
            parents.append(ci)
    return parents

# The output consumer and control_output ops are the children of this op
def _get_children_list(op):
    children = list(op._control_outputs)
    for output in op.outputs:
        children.extend(output.consumers())
    return children

# Prevent deadlocks caused by nccl ops not being scheduled in a consistent ordering across ranks.
def serialize_allreduce_ops(graph_targets, serialize_inputs=True, print_dag=False):

    # Traverse all graph_targets through their inputs and:
    # Build a mutable dag of dict()'s' with ops as keys and their input ops as values (as set() elements)
    # For ops with no inputs, add to the ready to scheudle list.
    dag     = dict()
    ready   = list()
    queue   = deque([t.op for t in graph_targets])
    visited = set()
    while queue:
        op = queue.popleft()
        if op not in visited:
            visited.add(op)
            inputs = _get_parents_set(op)
            if len(inputs):
                dag[op] = inputs
                # add parents to queue in deterministc order (not python set ordering)
                queue.extend(_get_parents_list(op))
            else:
                ready.append(op)

    # Implement topological sorting found here:
    # https://en.wikipedia.org/wiki/Topological_sorting
    # Pick out AllreduceNccl ops and append them to a list in the order we'd like them scheduled.
    waves    = list()
    nccl_ops = list()
    while len(ready):
        ready_new = list()
        for ready_op in ready:
            for child_op in _get_children_list(ready_op):
                child_inputs = dag.get(child_op)
                if child_inputs is not None:
                    if ready_op in child_inputs:
                        child_inputs.remove(ready_op)
                        if len(child_inputs) == 0:
                            ready_new.append(child_op)
                            dag.pop(child_op)
                            if child_op.type == "AllreduceNccl":
                                nccl_ops.append(child_op)
        waves.append(ready)
        ready = ready_new

    if len(dag):
        raise ValueError("Error: graph_targets have at least one cycle")

    # We could serialize all ops within each wave.
    # Instead, just serialize the ops that are the inputs to the nccl ops.
    # Don't serialize the nccl ops themselves since they are async.
    # We just need them to be scheduled in a consistent order.
    prev_op = None
    for nccl_op in nccl_ops:
        if serialize_inputs:
            input_op = nccl_op.inputs[0].op
            if prev_op is not None:
                input_op._add_control_input(prev_op)
            prev_op = input_op
        else:
            if prev_op is not None:
                nccl_op._add_control_input(prev_op)
            prev_op = nccl_op

    if print_dag:
        f = open(print_dag, 'w') if type(print_dag) is str else sys.stdout
        for wave in waves:
            for op in sorted(wave, key=lambda op: (op.type, op.name)):
                print(op.type, op.name, op.outputs[0].dtype, op.outputs[0].shape, file=f)
            print("", file=f)
        if f is not sys.stdout:
            f.close()


def identity_sync(*xs, sync_fwd=False, sync_bwd=True, name=None):
    ys = _op_module.identity_synchronize(xs, sync=sync_fwd, sync_bwd=sync_bwd, name=name)
    if len(ys) == 1:
        return ys[0]
    return ys

@ops.RegisterGradient("IdentitySynchronize")
def identity_sync_grad(op, *dys):
    if op.get_attr('sync_bwd'):
        return _op_module.identity_synchronize(dys, sync=True)
    return dys




##################### Simple nccl ops for sharding models accross gpus ##################################
# Uses a single comm
# Each MPI worker / Gpu can only be part of one fixed grouping.

init_group_size = None
init_group_indx = None
init_group_rank = None

def check_group_params(group_size, group_indx, group_rank):
    global init_group_size
    global init_group_indx
    global init_group_rank

    if init_group_size is None:
        init_group_size = group_size
    elif init_group_size != group_size:
        print(f"Warning: only the first value of group_size ({init_group_size}) that was passed in will be used.  group_size={group_size} value ignored.")

    if init_group_indx is None:
        init_group_indx = group_indx
    elif init_group_indx != group_indx:
        print(f"Warning: only the first value of group_indx ({init_group_indx}) that was passed in will be used.  group_indx={group_indx} value ignored.")

    if init_group_rank is None:
        init_group_rank = group_rank
    elif init_group_rank != group_rank:
        print(f"Warning: only the first value of group_rank ({init_group_rank}) that was passed in will be used.  group_rank={group_rank} value ignored.")


reduce_scatter_counter = 0

def reduce_scatter(x, group_size=1, group_indx=0, group_rank=0, transpose=True, name=None, debug_str=''):
    check_group_params(group_size, group_indx, group_rank)

    assert not x.device or x.device[-2:] == ":0", "Only one gpu per process currently supported by allreduce: " + x.device
    global reduce_scatter_counter

    if transpose:
        assert x.shape.ndims == 2, "input must be of dim 2 prior to reduce_scatter with transpose"
        x = _op_module.transpose2d(x)

    assert x.shape[0].value % group_size == 0, "leading dim must be multiple of group_size"

    y = _op_module.reduce_scatter_nccl(x,
        group_size = group_size,
        group_indx = group_indx,
        group_rank = group_rank,
        op_num     = reduce_scatter_counter,
        name       = name,
        debug_str  = debug_str)
    reduce_scatter_counter += 1

    if transpose:
        y = _op_module.transpose2d(y)

    return y


all_gather_counter = 0

def all_gather(x, group_size=1, group_indx=0, group_rank=0, transpose=True, name=None, debug_str=''):
    global all_gather_counter
    check_group_params(group_size, group_indx, group_rank)

    assert not x.device or x.device[-2:] == ":0", "Only one gpu per process currently supported by allreduce: " + x.device
    global reduce_scatter_counter

    if transpose:
        assert x.shape.ndims == 2, "input must be of dim 2 prior to all_gather with transpose"
        x = _op_module.transpose2d(x)

    y = _op_module.all_gather_nccl(x,
        group_size = group_size,
        group_indx = group_indx,
        group_rank = group_rank,
        op_num     = all_gather_counter,
        name       = name,
        debug_str  = debug_str)
    all_gather_counter += 1

    if transpose:
        y = _op_module.transpose2d(y)

    return y

@ops.RegisterGradient("ReduceScatterNccl")
def allreduce_grad(op, dy):

    global all_gather_counter

    dx = _op_module.all_gather_nccl(dy,
        group_size = op.get_attr('group_size'),
        group_indx = op.get_attr('group_indx'),
        group_rank = op.get_attr('group_rank'),
        debug_str  = op.get_attr('debug_str'),
        op_num     = all_gather_counter)

    all_gather_counter += 1
    return dx

@ops.RegisterGradient("AllGatherNccl")
def allreduce_grad(op, dy):

    global reduce_scatter_counter

    dx = _op_module.reduce_scatter_nccl(dy,
        group_size = op.get_attr('group_size'),
        group_indx = op.get_attr('group_indx'),
        group_rank = op.get_attr('group_rank'),
        debug_str=op.get_attr('debug_str'),
        op_num     = reduce_scatter_counter)

    reduce_scatter_counter += 1
    return dx

