#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from blocksparse import dw_matmul_large_n
import blocksparse.ewops as ew
from time import time

shapes = [
    [ 1024*1024, 32 ],
    [ 1024*128, 128 ],
    [ 1024*32,  512 ],
    [      32, 1024 ],
    [      64,    8 ],
    [      32,    4 ],
]

class MatMulTest(tf.test.TestCase):

    def testMatMul(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            for shape in shapes:

                np.random.seed(int(time()))
                cpuX = np.random.normal(loc=0.1, scale=1.0, size=shape).astype(np.float16).astype(np.float32)
                cpuE = np.random.normal(loc=0.2, scale=1.0, size=shape).astype(np.float16).astype(np.float32)
                cpuU = np.dot(cpuX.astype(np.float64).T, cpuE.astype(np.float64)).astype(np.float32)

                for dtype in (tf.float32, tf.float16):  #tf.float16, tf.bfloat16

                    with tf.device("/gpu:0"):

                        x = tf.placeholder(tf.float32, cpuX.shape, name="x")
                        e = tf.placeholder(tf.float32, cpuE.shape, name="e")

                        feed_dict = { x : cpuX, e : cpuE }

                        if dtype is not tf.float32:
                            xf = ew.float_cast(x, dtype=dtype)
                            ef = ew.float_cast(e, dtype=dtype)
                        else:
                            xf, ef = x, e

                        u0 = dw_matmul_large_n(xf, ef)
                        u1 = tf.matmul(xf, ef, transpose_a=True, transpose_b=False)

                        if dtype is not tf.float32:
                            u1 = ew.float_cast(u1, dtype=tf.float32, dx_dtype=dtype)

                        u0, u1 = sess.run( [ u0, u1 ], feed_dict )

                    for op, dev, cpu in [
                        ("custom", u0, cpuU),
                        ("cublas", u1, cpuU),
                    ]:

                        dif     = np.abs(cpu - dev)
                        avgval  = np.average(abs(cpu))
                        maxdif  = dif.max()
                        max_err = maxdif if avgval == 0 else maxdif / avgval
                        l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

                        print("%s, depth:%8d shape:%12s, op:%s, err:%17.12f, l2_err:%17.12f" % (dtype.name, shape[0], str(cpu.shape), op, maxdif, l2_err))

if __name__ == "__main__":
  tf.test.main()
