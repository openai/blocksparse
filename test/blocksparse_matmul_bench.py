#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from time import time

from blocksparse.matmul import BlocksparseMatMul, SparseProj, group_param_grads
import blocksparse.ewops as ew
import networkx

bench = 4000
depth = 8
mask  = "ba" # ba, ws

# multi-threading screws up benchmarking
conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

# bsize = 32
# for hsize in range(1,11):
#     n = hsize*80*32 // bsize
#     m = int(round((6553600 / (bsize*bsize*n) - 1) / 2))

#     print("%2d %5d %3d %2d %.2f" % (hsize, hsize*80*32, n, m, 100*(m*2 + 1) / n))
# exit()

with tf.Session(config=conf) as sess, tf.device("/gpu:0"):

    #for hsize, sparsity in ((1, 100.0), (2, 25.0), (3, 12.0), (4, 6.0), (6, 3.0)):

    # for hsize, sparsity in ((2, 100.0), (3, 43.0), (4, 24.0), (5, 15.0), (6, 10.5), (7, 8.0), (9, 4.6), (11, 3.0)): #(10, 3.7), (8, 6.0),
    #     hsize *= 56*32

    for hsize, sparsity in ( (1, 100.0), (2, 25.62), (3, 11.25), (4, 6.56), (5, 4.25), (6, 2.71), (7, 1.96), (8, 1.41) ): #(10, 3.7), (8, 6.0),
        hsize *= 80*32

        for bsize, axis in ( (32,0), (16,0), (8,0) ): # (32,0), (16,0), (8,0)

            n = hsize // bsize

            if sparsity == 100.0:
                layout = np.ones((n,n), dtype=np.int32)
                blks = n*n
                spar = sparsity
                m = n
            else:
                for m in range(1,n//2):
                    if mask == "ws":
                        blks = n * (m*2 + 1)
                    else:
                        blks = 2*m*(n-m) + m*m + n-m
                    spar = 100 * blks / n**2
                    if spar >= sparsity:
                        break

                if mask == "ws":
                    layout = networkx.generators.random_graphs.watts_strogatz_graph(n, m*2, .2)
                    layout = networkx.adjacency_matrix(layout).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
                else:
                    layout = networkx.generators.barabasi_albert_graph(n, m)
                    layout = networkx.adjacency_matrix(layout).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
                    layout[0:m,0:m] = 1

            # print("axis:%d bsize:%2d hsize:%d params:%d sparsity:%.2f m:%d" % (axis, bsize, hsize, bsize*bsize*blks, spar, m))
            # continue

            bsmm = BlocksparseMatMul(layout, block_size=bsize, feature_axis=axis, name="test")

            W = np.random.uniform(-1.0, 1.0, bsmm.w_shape).astype(np.float32)
            w = tf.constant(W)

            for N in (64,): # 128,64,32,16,1,

                X = np.random.uniform(-1.0, 1.0, bsmm.i_shape(N)).astype(np.float32)
                E = np.random.uniform(-1.0, 1.0, bsmm.o_shape(N)).astype(np.float32)
                x = tf.constant(X)
                e = tf.constant(E)

                for dtype in (tf.bfloat16, ): # tf.bfloat16, tf.bfloat32,

                    #print("axis:%d bsize:%2d N:%d dtype:%s hsize:%d params:%d sparsity:%.2f" % (axis, bsize, N, dtype.name, hsize, bsize*bsize*blks, spar))

                    # compute in tensorflow
                    w2 = ew.float_cast(w, dtype=dtype)
                    y  = ew.float_cast(x, dtype=dtype)

                    for j in range(depth):
                        repeat = bench if bench and j==depth-1 else 0
                        y = bsmm(y, w2, dw_dtype=dtype, bench=repeat) # (bench and j==depth-1) (bench and j==0)

                    y = ew.float_cast(y, dtype=tf.float32, dx_dtype=dtype)
                    #sess.run( y )

                    d = tf.gradients(y, [x, w], e, aggregation_method=3)
                    if depth > 1:
                        d[1] = group_param_grads(d[1], 8)

                    y, (dx, dw) = sess.run( [y, d] )

