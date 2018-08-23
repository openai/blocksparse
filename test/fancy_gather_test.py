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
    [ 128, 16, 149,     ],
    [ 128, 16,  30,     ], # int32
    [ 128, 16,  21,     ],
    [ 128, 16,   9,     ],
    [ 128, 16,   4,     ],
    [ 128, 16,   5, 128 ],
    [ 128, 16,   6, 128 ],
    [ 128, 16,  62, 128 ],
]

class FancyGatherTest(tf.test.TestCase):

    def testFancyGather(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            for shape in shapes:

                idx_shape = shape[0:2]
                idx_dim   = shape[2]
                out_shape = idx_shape + shape[3:]

                for dtype in (tf.float32, ):  #tf.float16, tf.bfloat16

                    #rtol = 1e-4 if dtype is tf.float32 else 1e-1

                    #tf.reset_default_graph()
                    np.random.seed(int(time()))
                    cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                    cpuA = np.random.randint(0, idx_dim, size=idx_shape, dtype=np.int32)
                    cpuE = np.random.uniform(-1.0, 1.0, out_shape).astype(np.float32)

                    with tf.device("/gpu:0"):

                        x = tf.placeholder(tf.float32, cpuX.shape)
                        a = tf.placeholder(tf.int32,   cpuA.shape)
                        e = tf.placeholder(tf.float32, cpuE.shape)

                        feed_dict = { x: cpuX, a: cpuA, e: cpuE }

                        xf = ew.float_cast(x, dtype=dtype)
                        y  = ew.float_cast(ew.fancy_gather(xf, a), dtype=tf.float32, dx_dtype=dtype)

                        devY, (devB,) = sess.run( [y, tf.gradients(y, [x], e)], feed_dict )

                        y = ew.fancy_gather(x, a, use_tf=True)

                        cpuY, (cpuB,) = sess.run( [y, tf.gradients(y, [x], e)], feed_dict )

                    for op, devT, cpuT in (
                        ( "devY", devY, cpuY ),
                        ( "devB", devB, cpuB )):

                        difA   = np.abs(cpuT - devT)
                        maxdif = difA.max()
                        sumerr = (difA > .001).sum()
                        poserr = np.argmax(np.abs(difA).reshape(-1))

                        print("%s, shape:%22s, op:%s, err:%17.12f, sum_err: %d, pos_err:%d" % (dtype.name, str(shape), op, maxdif, sumerr, poserr))


if __name__ == "__main__":
  tf.test.main()
