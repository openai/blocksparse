#!/usr/bin/env python

# nvprof -f -o "nccl_test_%p.nvvp" --profile-child-processes
# nvprof --profile-child-processes

import numpy as np
import platform
from collections import defaultdict
from mpi4py import MPI
import blocksparse.nccl as nccl
import blocksparse.ewops as ew
from time import time
import tensorflow as tf
import os

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

print("starting process %d mpi size %d" % (mpi_rank, mpi_size), flush=True)

config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="%d" % (mpi_rank % 8)))

with tf.Session(config=config) as sess, tf.device("/gpu:0"):

    N = 1024*4
    shape = (N,N)
    prereduce = True

    np.random.seed(1)
    A = np.random.normal(loc=0.1, scale=1.0, size=shape).astype(np.float32)
    B = np.random.normal(loc=0.2, scale=1.0, size=shape).astype(np.float32)
    a = tf.placeholder(tf.float32, A.shape, name="a")
    b = tf.placeholder(tf.float32, B.shape, name="b")
    feed_dict = { a : A, b : B }

    prereduce = min(mpi_size, 8) if prereduce else 0

    for dtype in (tf.float32, tf.float16):  #tf.float16, tf.bfloat16

        y0 = tf.matmul(a, b)
        y0 = ew.float_cast(y0, dtype=dtype)
        y0 = nccl.allreduce(y0, rank=mpi_rank, num_comms=1, prereduce=prereduce)
        y0 = ew.float_cast(y0, dtype=tf.float32)

        y0 = sess.run(y0, feed_dict=feed_dict)

        if mpi_rank == 0:
            y1 = np.dot(A, B) * mpi_size

            dif     = np.abs(y1 - y0)
            avgval  = np.average(abs(y1))
            maxdif  = dif.max()
            max_err = maxdif if avgval == 0 else maxdif / avgval
            l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(y1).sum())

            print("prereduce: %d, dtype: %s, shape:%12s, err:%17.12f, l2_err:%17.12f" % (prereduce, dtype.name, str(shape), maxdif, l2_err))

