#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function
from blocksparse.embed import embedding_lookup
import blocksparse.ewops as ew
from time import time

shapes = [
    [ [  39,  16], [512, 16, 149]   ],
    [ [   5,  64], [512, 16, 1]     ],
    [ [   4,  64], [512, 16, 1]     ],
    [ [  93,  16], [512, 16, 139]   ],
    [ [ 268,   8], [512, 16, 6]     ],
    [ [1506,  16], [512, 16, 100]   ],
    [ [ 723,  32], [512, 16, 60]    ],
    [ [ 260,  32], [512, 16, 150]   ],
    [ [  19, 256], [512, 16, 5]     ],
    [ [ 657,  64], [512, 16, 5, 30] ],
    [ [ 657, 128], [512, 16, 5, 30] ],
    [ [  1,  1], [1]   ],
    [ [  32*1024,  1024], [ 1, 1024]   ],
    [ [  32*1024,  1024], [ 8, 1024]   ],
    [ [  32*1024,  1024], [16, 1024]   ],
    [ [  32*1024,  1024], [32, 1024]   ],
]

bench = 0

class EmbeddingLookupTest(tf.test.TestCase):

    def testEmbeddingLookup(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

                for shapeW, shapeI in shapes:

                    C = shapeW[0]
                    shapeY = shapeI + shapeW[1:]

                    np.random.seed(int(time()))
                    cpuI = np.random.randint(0, C, size=shapeI, dtype=np.int32)
                    cpuW = np.random.uniform(-1.0, 1.0, shapeW).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shapeY).astype(np.float32)

                    for dtype in (tf.float32, tf.float16, ):  #tf.float16, tf.float32
                        for sort in (True, False):

                            results = []
                            for device in ("gpu", "cpu"):

                                if bench and device == "cpu":
                                    break

                                castW = device == "gpu" and dtype is not tf.float32
                                if castW:
                                    if C <= 256:
                                        castI = tf.uint8
                                    elif C <= 65536:
                                        castI = tf.uint16
                                    else:
                                        castI = None
                                else:
                                    castI = None


                                with tf.device("/%s:0" % device), tf.name_scope(device):

                                    i = tf.placeholder(tf.int32,   cpuI.shape, name="i")
                                    w = tf.placeholder(tf.float32, cpuW.shape, name="w")
                                    e = tf.placeholder(tf.float32, cpuE.shape, name="e")

                                    feed_dict = { i : cpuI, w : cpuW, e : cpuE }

                                    wf = ew.float_cast(w, dtype=dtype) if castW else w
                                    i  = tf.cast(i, dtype=castI) if castI is not None else i

                                    y = embedding_lookup(wf, i, sort_grad=sort, bench=bench)

                                    if castW:
                                        y = ew.float_cast(y, dtype=tf.float32)

                                    dw, = tf.gradients(y, [w], e)

                                    results.append( sess.run( [ y, dw ], feed_dict ) )

                            if not bench:

                                for op, dev, cpu in zip(["y", "dw"], results[0], results[1]):

                                    dif     = np.abs(cpu - dev)
                                    avgval  = np.average(abs(cpu))
                                    maxdif  = dif.max()
                                    max_err = maxdif if avgval == 0 else maxdif / avgval
                                    l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

                                    print("%s, shape:%22s, op:%3s, err:%17.12f, l2_err:%17.12f" % (dtype.name, str(cpu.shape), op, max_err, l2_err))



if __name__ == "__main__":
  tf.test.main()


# @function.Defun(
#     python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
#     shape_func=lambda op: [op.inputs[0].get_shape()])
# def convert_gradient_to_tensor(x):
#     return x

# np_x = np.random.randint(0, 64, size=[512, 16, 150], dtype=np.int32)
# np_w = np.random.uniform(-1.0, 1.0, [64, 32]).astype(np.float32)
# np_e = np.random.uniform(-1.0, 1.0, [512, 16, 150, 32]).astype(np.float32)

# with tf.Session() as sess, tf.device("/gpu:0"):

#     x = tf.placeholder(tf.int32,   np_x.shape, name="x")
#     w = tf.placeholder(tf.float32, np_w.shape, name="w")
#     e = tf.placeholder(tf.float32, np_e.shape, name="e")

#     feed_dict = {
#         x : np_x,
#         w : np_w,
#         e : np_e,
#     }

#     #y = tf.nn.embedding_lookup(w, x)

#     wf = ew.float_cast(w, dtype=tf.float16)

#     y = tf.gather(convert_gradient_to_tensor(wf), x)

#     y = ew.float_cast(y, dtype=tf.float32)

#     dw = tf.gradients(y, [w], e)[0]

#     y, dw = sess.run([y, dw], feed_dict)
