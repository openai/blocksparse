#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow  as tf
import blocksparse as bs
from tensorflow.python.ops import gradient_checker

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

np.set_printoptions(linewidth=600, formatter={'float':lambda x: "%.2f" % x})

class PruneTest(tf.test.TestCase):

    def testPrune(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            with tf.device("/cpu:0"):
                sparse_ph = tf.placeholder(tf.float32, shape=[])
                step_ph   = tf.placeholder(tf.int32,   shape=[])

            blocks   = 1000
            sparsity = 0.1
            for ntype in ("max", "l2"):
                for bsize in (8,16,32):


                    W = np.random.normal(0.0, 1.0, (blocks,bsize,bsize)).astype(np.float32)
                    G = np.ones((blocks,), dtype=np.float32)

                    w = tf.get_variable(f"w_{ntype}_{bsize}", initializer=W)
                    g = tf.get_variable(f"g_{ntype}_{bsize}", initializer=G)

                    sess.run( tf.global_variables_initializer() )

                    prune_op = bs.blocksparse_prune(w, g, step_ph, sparsity=sparse_ph, norm=ntype, frequency=1)
                    norm_op  = bs.blocksparse_norm(w, norm=ntype)

                    sess.run([prune_op], feed_dict={ sparse_ph: sparsity, step_ph: 0 })
                    n, g = sess.run([norm_op, g])

                    if ntype == "max":
                        N = np.max(np.abs(W.reshape(blocks,-1)), axis=1)
                    else:
                        N = np.sqrt(np.sum(np.square(W.reshape(blocks,-1)), axis=1))

                    keep = int(round(blocks * (1.0 - sparsity)))
                    for si, (v, i) in enumerate(sorted(list(zip(N, range(blocks))), reverse=True)):
                        if si >= keep:
                            G[i] = 0.0

                    print("Type: %3s bsize: %2d norm_err: %.5f gate_err: %.0f" % ( ntype, bsize, np.sum(np.abs(N - n)), np.sum(np.abs(G - g)) ))
                    # print("N", N)
                    # print("n", n)
                    # print("G", G)
                    # print("g", g)

    def atestGateGrad(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            dtype = tf.float16

            layout = np.ones([2,2], dtype=np.bool)
            bsmm   = bs.BlocksparseMatMul(layout, block_size=8, feature_axis=0, name="test")

            X = np.random.uniform(-1.0, 1.0, bsmm.i_shape(64)).astype(np.float16).astype(np.float32)
            W = np.random.uniform(-1.0, 1.0, bsmm.w_shape    ).astype(np.float16).astype(np.float32)
            G = np.random.uniform( 0.0, 1.0, bsmm.blocks     ).astype(np.float16).astype(np.float32)
            #G = np.ones([bsmm.blocks], dtype=np.float32)

            x = tf.constant(X)
            w = tf.constant(W)
            g = tf.constant(G)

            wf = bs.float_cast(w, dtype=dtype)
            xf = bs.float_cast(x, dtype=dtype)

            y = bsmm(xf, wf, gate=g, gate_grad=True, bench=0)

            y = bs.float_cast(y, dtype=tf.float32)

            sess.run( tf.global_variables_initializer() )

            # y = sess.run( y )
            # exit()

            error = gradient_checker.compute_gradient_error(x, x.shape, y, y.shape) #, extra_feed_dict={ x: cpuX, m: mask }
            print(error)

            error = gradient_checker.compute_gradient_error(w, w.shape, y, y.shape) #, extra_feed_dict={ x: cpuX, m: mask }
            print(error)

            error = gradient_checker.compute_gradient_error(g, g.shape, y, y.shape) #, extra_feed_dict={ x: cpuX, m: mask }
            print(error)

            #assert error < 0.01, error


if __name__ == "__main__":
  tf.test.main()


# with tf.Session() as sess:

#     with tf.device("/gpu:0"):

#         xshape = [25528,] #25528

#         X = np.random.uniform(-1.0, 1.0, xshape).astype(np.float32)
#         x = tf.placeholder(tf.float32, X.shape)

#         val, idx = tf.nn.top_k(X, k=X.size, sorted=True)

#         val, idx = sess.run( [val, idx], feed_dict={ x: X })

#         print(X[0])
#         print(val[0])
#         print(idx[0])

# CN
# MV = (1, C)
# GB = (C, 1)

# NC
# MV = (1, C)
# GB = (1, C)