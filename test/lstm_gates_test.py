#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
import blocksparse.norms as norms
import blocksparse.lstm as lstm
from time import time

shapes = [
    [ 128, 1024*1 ],
    [ 128, 1024*2 ],
]

layernorm = False

class LSTMGatesTest(tf.test.TestCase):

    def testLSTMGates(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            for shape1 in shapes:
                shape4 = [shape1[0], shape1[1]*4]

                for dtype in (tf.float32, tf.float16):  #tf.float16, tf.bfloat16

                    np.random.seed(int(time()))
                    cpuC = np.random.uniform(-1.0, 1.0, shape1    ).astype(np.float32)
                    cpuH = np.random.uniform(-1.0, 1.0, shape4    ).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape1    ).astype(np.float32)
                    cpuB = np.random.uniform(-1.0, 1.0, shape4[1:]).astype(np.float32)
                    cpuG = np.random.uniform(-1.0, 1.0, shape4[1:]).astype(np.float32)

                    results = []
                    for device in ("gpu", "cpu"):

                        with tf.device("/%s:0" % device), tf.name_scope(device):

                            c = tf.placeholder(tf.float32, cpuC.shape, name="c")
                            h = tf.placeholder(tf.float32, cpuH.shape, name="h")
                            e = tf.placeholder(tf.float32, cpuE.shape, name="e")
                            b = tf.placeholder(tf.float32, cpuB.shape, name="b")
                            g = tf.placeholder(tf.float32, cpuB.shape, name="g")

                            feed_dict = {
                                c : cpuC,
                                h : cpuH,
                                e : cpuE,
                                b : cpuB,
                                g : cpuG,
                            }

                            if device == "gpu" and dtype is not tf.float32:
                                cf = ew.float_cast(c, dtype=dtype)
                                hf = ew.float_cast(h, dtype=dtype)
                            else:
                                cf, hf = c, h

                            if layernorm:
                                hf = norms.layer_norm(hf, g, b, axis=1, segments=4)
                                bias = None
                            else:
                                bias = b

                            cf, hf = lstm.fused_lstm_gates(cf, hf, bias=bias, forget_bias=1.0)

                            if device == "gpu" and dtype is not tf.float32:
                                cf = ew.float_cast(cf, dtype=tf.float32, dx_dtype=dtype)
                                hf = ew.float_cast(hf, dtype=tf.float32, dx_dtype=dtype)

                            if layernorm:
                                dc, dh, dg, db = tf.gradients([cf, hf], [c, h, g, b], [None, e])
                                results.append( sess.run( [ cf, hf, dc, dh, dg, db ], feed_dict ) )
                                labels = [" c", " h", "dc", "dh", "dg", "db"]
                            else:
                                dc, dh, db = tf.gradients([cf, hf], [c, h, b], [None, e])
                                results.append( sess.run( [ cf, hf, dc, dh, db ], feed_dict ) )
                                labels = [" c", " h", "dc", "dh", "db"]


                    for op, dev, cpu in zip(labels, results[0], results[1]):

                        dif     = np.abs(cpu - dev)
                        avgval  = np.average(abs(cpu))
                        maxdif  = dif.max()
                        max_err = maxdif if avgval == 0 else maxdif / avgval
                        l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())


                        print("%s, shape:%12s, op:%s, err:%17.12f, l2_err:%17.12f" % (dtype.name, str(cpu.shape), op, maxdif, l2_err))


if __name__ == "__main__":
  tf.test.main()
