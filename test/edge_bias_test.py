#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from operator import mul

from blocksparse.conv import ConvEdgeBias, ceil_div
import blocksparse.ewops as ew

ones = 0
out  = 0
bench = 0

shapes = [
    #   N    K    RS       HW     strides
    [   1, 512, [3,3], [128,128], [1,1] ],
    [   1, 512, [3,3], [ 64, 64], [1,1] ],
    [   1, 512, [3,3], [ 32, 32], [1,1] ],
    [   1, 512, [3,3], [ 16, 16], [1,1] ],
    [   1, 512, [3,3], [  8,  8], [1,1] ],
    [   1, 512, [3,3], [  4,  4], [1,1] ],

    [   1,   6, [3,3], [128,128], [1,1] ],
    [   1,  12, [3,3], [ 64, 64], [1,1] ],
    [   1,  24, [3,3], [ 32, 32], [1,1] ],
    [   1,  48, [3,3], [ 16, 16], [1,1] ],
    [   1,  96, [3,3], [  8,  8], [1,1] ],
    [   1, 192, [3,3], [  4,  4], [1,1] ],
]



class EdgeBiasTest(tf.test.TestCase):

    def testEdgeBias(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            test = 0
            for N, K, RS, HW, strides in shapes:
                test += 1
                PQ = [ ceil_div(x, std) for x, std in zip(HW, strides) ]

                for layout in ("NCHW","NHWC",): # "NCHW","NHWC"

                    if layout == "NHWC":
                        y_shape = [N] + PQ + [K]
                        x_shape = [N] + HW + [K]
                        w_shape = RS + [K, K]
                    else:
                        y_shape = [N] + [K] + PQ
                        x_shape = [N] + [K] + HW
                        w_shape = [K, K] + RS

                    eb = ConvEdgeBias(y_shape, x_shape, w_shape, strides=strides, data_format=layout)

                    if ones:
                        cpuX = np.ones(y_shape).astype(np.float32)
                        cpuE = np.ones(y_shape).astype(np.float32)
                        cpuG = np.ones(eb.shape).astype(np.float32)
                        cpuB = np.ones(eb.shape).astype(np.float32)
                    else:
                        cpuX = np.random.uniform(-1.0, 1.0, y_shape).astype(np.float32)
                        cpuE = np.random.uniform(-1.0, 1.0, y_shape).astype(np.float32)
                        cpuG = np.random.uniform(-1.0, 1.0, eb.shape).astype(np.float32)
                        cpuB = np.random.uniform(-1.0, 1.0, eb.shape).astype(np.float32)

                    x = tf.placeholder(tf.float32, cpuX.shape)
                    e = tf.placeholder(tf.float32, cpuE.shape)
                    g = tf.placeholder(tf.float32, cpuG.shape)
                    b = tf.placeholder(tf.float32, cpuB.shape)

                    feed_dict = { x: cpuX, e: cpuE, g:cpuG, b:cpuB }

                    for dtype in (tf.float32,):  # tf.float32, tf.float16, tf.bfloat16

                        xf = ew.float_cast(x, dtype=dtype)
                        y = eb(xf, g, b, bench=bench)
                        y = ew.float_cast(y, dtype=tf.float32, dx_dtype=dtype)

                        devY, (devDX, devDG, devDB) = sess.run( [y, tf.gradients(y, [x, g, b], e)], feed_dict )

                        if bench == 0:

                            cpuY = eb.edge_bias_test(cpuX, cpuG, cpuB)
                            cpuDX, cpuDG, cpuDB = eb.edge_bias_grad_test(cpuE, cpuX, cpuG)

                            for op, devT, cpuT in (
                                ( " devY", devY,  cpuY  ),
                                ( "devDX", devDX, cpuDX ),
                                ( "devDG", devDG, cpuDG ),
                                ( "devDB", devDB, cpuDB ),):

                                devT = np.array(devT)
                                difA = cpuT - devT

                                avgval = abs(cpuT).sum() / cpuT.size
                                maxdif = abs(difA).max()
                                ratio  = maxdif / avgval

                                print("%8s, test:%2d layout: %s op:%s err:%17.12f" % (dtype.name, test, layout, op, ratio))


if __name__ == "__main__":
  tf.test.main()
