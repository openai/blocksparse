#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from time import time
from blocksparse.conv  import cwise_linear
from blocksparse.ewops import float_cast

ones = 0
out  = 0
shapes = [
    [  1, 32, 32        ],
    [ 64, 64, 32        ],
    [  8, 64,  4,  4    ],
    [  8, 64, 16, 16    ],
    [  8, 64, 32, 32    ],
    [  8, 64,  8,  8, 8 ],
]

class CWiseLinearTest(tf.test.TestCase):

    def testCWiseLinear(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        with self.test_session(config=config) as sess:

            for shape in (shapes):

                bshape    = [1] * len(shape)
                bshape[1] = shape[1]

                if ones:
                    cpuX = np.ones(shape,  dtype=np.float32)
                    cpuE = np.ones(shape,  dtype=np.float32)
                    cpuG = np.ones(bshape, dtype=np.float32)
                    cpuB = np.ones(bshape, dtype=np.float32)
                else:
                    np.random.seed(int(time()))
                    cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                    cpuG = np.random.uniform(-1.0, 1.0, bshape).astype(np.float32)
                    cpuB = np.random.uniform(-1.0, 1.0, bshape).astype(np.float32)

                for dtype in (tf.float32, tf.float16, ):  # tf.float32, tf.float16, tf.bfloat16
                    relus = (True, False) if dtype is tf.float32 else (False,)
                    for relu in relus:

                        results = []
                        for device in ("gpu", "cpu"):

                            cast = device == "gpu" and dtype is not tf.float32

                            with tf.device("/%s:0" % device), tf.name_scope(device):

                                x = tf.placeholder(tf.float32, cpuX.shape, name="x")
                                e = tf.placeholder(tf.float32, cpuE.shape, name="e")
                                g = tf.placeholder(tf.float32, cpuG.shape, name="g")
                                b = tf.placeholder(tf.float32, cpuB.shape, name="b")

                                feed_dict = {
                                    x : cpuX,
                                    e : cpuE,
                                    g : cpuG,
                                    b : cpuB,
                                }

                                xf = float_cast(x, dtype=dtype) if cast else x

                                y0 = cwise_linear(xf, gain=g, bias=b, relu=relu)
                                y1 = cwise_linear(xf, gain=g,         relu=relu)
                                y2 = cwise_linear(xf,         bias=b, relu=relu)

                                if cast:
                                    y0 = float_cast(y0, dtype=tf.float32)
                                    y1 = float_cast(y1, dtype=tf.float32)
                                    y2 = float_cast(y2, dtype=tf.float32)

                                dx0, dg0, db0 = tf.gradients(y0, [ x, g, b ], e)
                                dx1, dg1      = tf.gradients(y1, [ x, g    ], e)
                                dx2,      db2 = tf.gradients(y2, [ x,    b ], e)

                                results.append( sess.run( [ y0, y1, y2, dx0, dg0, db0, dx1, dg1, dx2, db2 ], feed_dict ) )
                                labels = ["y0", "y1", "y2", "dx0", "dg0", "db0", "dx1", "dg1", "dx2", "db2"]

                        for op, dev, cpu in zip(labels, results[0], results[1]):

                            dif     = np.abs(cpu - dev)
                            avgval  = np.average(abs(cpu))
                            maxdif  = dif.max()
                            max_err = maxdif if avgval == 0 else maxdif / avgval
                            l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

                            print("%s, shape:%16s, op: %3s, relu:%d, err:%17.12f, l2_err:%17.12f" % (dtype.name, str(cpu.shape), op, int(relu), max_err, l2_err))

                            # if out:
                            #     np.savetxt("out.txt",  difA.reshape(reshape), fmt='%5.2f')
                            #     np.savetxt("outC.txt", cpuT.reshape(reshape), fmt='%5.2f')
                            #     np.savetxt("outD.txt", devT.reshape(reshape), fmt='%5.2f')
                            #     exit()



if __name__ == "__main__":
  tf.test.main()
