#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
from time import time

shapes = [
    # [64, 16, 10,   10, 16, ],
    # [64, 16, 10,    6, 32, ],
    # [64, 16, 10,   15, 32, ],
    # [64, 16,  5,  256,     ],
    # [64, 16,  6,   32,     ],
    # [64, 16, 15,   64,     ],
    # [64, 16, 51,   64,     ],
    # [64, 16,256,   64,     ],
    [ 128, 16, 10,  10, 32 ],
    [ 128, 16, 10,  15, 64 ],
    [ 128, 16, 10,   6, 64 ],
    [ 128, 16, 15, 128,    ],
    [ 128, 16,  5, 512,    ],
    [ 128, 16,  6,  32,    ],
    [ 128, 16, 62, 128,    ],
]

class BiasReluTest(tf.test.TestCase):

    def testBiasRelu(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

                for shapeX in shapes:
                    axis   = len(shapeX)-2
                    shapeY = list(shapeX)
                    shapeY[axis] = 1

                    np.random.seed(int(time()))
                    cpuX = np.random.uniform(-2**14, 2**14, shapeX).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-2**14, 2**14, shapeY).astype(np.float16).astype(np.float32)

                    for dtype in (tf.float16, ):  #tf.float16, tf.float32

                        results = []
                        for device in ("gpu", "cpu"):

                            cast = device == "gpu" and dtype is not tf.float32

                            with tf.device("/%s:0" % device), tf.name_scope(device):

                                x = tf.placeholder(tf.float32, cpuX.shape, name="x")
                                e = tf.placeholder(tf.float32, cpuE.shape, name="e")

                                feed_dict = { x : cpuX, e : cpuE }

                                xf = ew.float_cast(x, dtype=dtype) if cast else x

                                y = ew.reduce_max(xf, axis=axis, keepdims=True)

                                if cast:
                                    y = ew.float_cast(y, dtype=tf.float32)

                                dx, = tf.gradients(y, [x], e)

                                results.append( sess.run( [ y, dx ], feed_dict ) )

                        for op, dev, cpu in zip(["y", "dx"], results[0], results[1]):

                            dif     = np.abs(cpu - dev)
                            sum_err = (dif > .01).sum()
                            pct_err = 100*sum_err / cpu.size
                            l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

                            print("%s, shape:%22s, op:%3s, sum_err: %4d, pct_err: %.4f, l2_err:%17.12f" % (dtype.name, str(cpu.shape), op, sum_err, pct_err, l2_err))


if __name__ == "__main__":
  tf.test.main()
