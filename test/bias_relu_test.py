#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from operator import mul

import blocksparse.ewops as ew
from tensorflow.python.framework import function

ones = 0
out  = 0
bench = 0
atomics = False
shapes = [
    (1, 1),
    (32, 32),

    (64 ,8192),
    (64 ,4096),
    (64 ,2048),
    (64 ,1024),

    (2**5 ,8193),
    (2**6 ,4097),
    (2**7 ,2049),
    (2**8 ,1025),
]

class BiasReluTest(tf.test.TestCase):

    def testBiasRelu(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:
            for shape in shapes:
                for axis in (0,1):
                    if axis == 0:
                        xshape = tuple(reversed(shape))
                        bshape = (shape[1], 1)
                    else:
                        xshape = shape
                        bshape = (1, shape[1])

                    if ones:
                        cpuX = np.ones(xshape, dtype=p.float32)
                        cpuE = np.ones(xshape, dtype=p.float32)
                        cpuB = np.ones(bshape, dtype=p.float32)
                    else:
                        cpuX = np.random.uniform(-1.0, 1.0, xshape).astype(np.float16).astype(np.float32)
                        cpuE = np.random.uniform(-1.0, 1.0, xshape).astype(np.float16).astype(np.float32)
                        cpuB = np.random.uniform(-1.0, 1.0, bshape).astype(np.float32)

                    for relu in (True, False):
                        for dtype in (tf.float16, tf.float32):  #tf.float16, tf.bfloat16

                            results = []
                            for device in ("gpu", "cpu"):
                                if bench and device == "cpu":
                                    break

                                cast = device == "gpu" and dtype is not tf.float32

                                with tf.device("/%s:0" % device), tf.name_scope(device):

                                    x = tf.placeholder(tf.float32, cpuX.shape)
                                    e = tf.placeholder(tf.float32, cpuE.shape)
                                    b = tf.placeholder(tf.float32, cpuB.shape)

                                    feed_dict = { x: cpuX, e: cpuE, b:cpuB }

                                    xc = ew.float_cast(x, dtype=dtype) if cast else x

                                    print(axis, xc.shape, b.shape)

                                    y = ew.bias_relu(xc, b, axis=axis, fast_gelu=relu, atomics=atomics, bench=bench)

                                    if cast:
                                        y = ew.float_cast(y, dtype=tf.float32)

                                    dx, db = tf.gradients(y, [x, b], e)

                                    results.append( sess.run( [ y, dx, db ], feed_dict ) )

                            if not bench:
                                for op, dev, cpu in zip(["y", "dx", "db"], results[0], results[1]):

                                    dif     = np.abs(cpu - dev)
                                    avgval  = np.average(abs(cpu))
                                    maxdif  = dif.max()
                                    max_err = maxdif if avgval == 0 else maxdif / avgval
                                    l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

                                    print("%s, shape:%14s, op:%3s(%d), err:%17.12f, l2_err:%17.12f" % (dtype.name, str(cpu.shape), op, relu, maxdif, l2_err))



if __name__ == "__main__":
  tf.test.main()

