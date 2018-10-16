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
    #(32, 32),

    (2**18, 256),
    (2**19, 128),
    (2**20,  64),
    (2**20,  32),
    (2**20,  16),
    (2**20,   8),
    (2**20,   4),

    (2**18, 257),
    (2**19, 129),
    (2**20,  65),
    (2**20,  33),
    (2**20,  17),
    (2**20,   9),
    (2**20,   5),
    (2**20,   1),

    (64 ,8192),
    (64 ,4096),
    (64 ,2048),
    (64 ,1024),

    # (2**5 ,8193),
    # (2**6 ,4097),
    # (2**7 ,2049),
    # (2**8 ,1025),

    # (204800, 16),
    # (122880, 32),
    # (307200, 32),
    # (10240, 256),
    # (10240, 256),
    # (104448, 64),
    # (104448, 64),
    # (12288,  32),
    # (30720,  64),
    # (2048, 1024),
    # (2048,   64),
    # (2048,   32),
    # (2048,    9),
    # (2048,    4),
    # (2048,    1),

    # [24,1024*8],
    # [120,  512],
    # [144,   32],
    # [1440,  64],
    # [1488, 128],
    # [24,     1],
    # [24,     4],
    # [24,     9],
    # [24,    21],
    # [24,    64],
    # [24,   768],
    # [24,  2048],
    # [2400,  32],
    # [360,  128],
    # [3600,  64],
    # [600,    4],
]

class BiasReluTest(tf.test.TestCase):

    def testBiasRelu(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:
            for shape in shapes:

                # shape[0] //= 24
                # shape[0]  *= 512

                if ones:
                    cpuX = np.ones(shape, dtype=p.float32)
                    cpuE = np.ones(shape, dtype=p.float32)
                    cpuB = np.ones(shape[1:], dtype=p.float32)
                else:
                    cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuB = np.random.uniform(-1.0, 1.0, shape[1:]).astype(np.float32)

                for relu in (True, False):
                    for dtype in (tf.float32, ):  #tf.float16, tf.bfloat16

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

                                y = ew.bias_relu(xc, b, relu=relu, atomics=atomics, bench=bench)

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

