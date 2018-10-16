
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from blocksparse.ewops import float_cast

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))


############################## Nccl Ops #####################################

op_counter = 0
init_num_comms = None
init_prereduce = None

def allreduce(x, sync_size=0, num_comms=2, logfile="", rank=0, prereduce=0, name=None):
    assert not x.op.device or x.op.device[-2:] == ":0", "Only one gpu per process currently supported by allreduce: " + x.op.device
    global op_counter
    global init_num_comms
    global init_prereduce

    if init_num_comms is None:
        init_num_comms = num_comms
    elif init_num_comms != num_comms:
        print("Warning: only the first value of num_comms (%d) that was passed in will be used.  num_comms=%d vale ignored." % (init_num_comms, num_comms))

    if init_prereduce is None:
        init_prereduce = prereduce
    elif init_prereduce != prereduce:
        print("Warning: only the first value of prereduce (%d) that was passed in will be used.  prereduce=%d vale ignored." % (init_prereduce, prereduce))

    if logfile and rank == 0:
        print("%03d %s" % (op_counter, x.name))
    ret = _op_module.allreduce_nccl(x, op_num=op_counter, sync_size=sync_size, num_comms=num_comms, prereduce=prereduce, logfile=logfile, name=name)
    op_counter += 1
    return ret


def group_allreduce(grads, parms, search_strings=None, cast_map=None, cast_all=None, num_comms=2, prereduce=0):

    # if no grouping specified, create one group to reduce at the end (no overlap with compute)
    if search_strings is None:
        search_strings = ["group_allreduce_all"]

    groups = [(name, list(), list()) for name in search_strings]

    for i, (grad, param) in enumerate(zip(grads, parms)):
        for name, group16, group32 in groups:
            if name == search_strings[-1] or name in param.name:

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
        for group in (group16, group32):
            count += len(group)
            if len(group) > 0:
                if len(group) == 1:
                    concated = group[0][1]
                else:
                    concated = tf.concat([tf.reshape(grad, [-1]) for _, grad, _ in group], 0, name="concat_"+name)

                reduced = allreduce(concated, num_comms=num_comms, prereduce=prereduce)

                if len(group) == 1:
                    grads[group[0][0]] = reduced
                else:
                    offset = 0
                    for i, grad, param in group:
                        size     = param.shape.num_elements()
                        grads[i] = tf.reshape(reduced[offset: offset + size], param.shape)
                        offset  += size

        if count == 0:
            print("Warning: no grads found for all_reduce group: ", name)

    # nothing to return, grads modified in place

def sync_variables_op(mpi_rank, num_comms=2, prereduce=0):
    ops  = list()
    prev = []
    with tf.device("/gpu:0"):
        for var in tf.trainable_variables():
            with tf.control_dependencies(prev):
                op = tf.assign(var, allreduce(var if mpi_rank == 0 else var * 0.0, num_comms=num_comms, prereduce=prereduce))
            prev = [op]
            ops.append(op)

    return tf.group(*ops)


