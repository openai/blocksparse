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

np.set_printoptions(threshold=8193, linewidth=600, formatter={'int':lambda x: "%4d" % x,'float':lambda x: "%8.6f" % x})

dtypes = [
    (tf.float32,  tf.float32),
    # (tf.float16,  tf.float16),
    #(tf.bfloat16, tf.bfloat16),
    #(tf.float16,  tf.float32),
    #(tf.bfloat16, tf.float32),
]

one = 0
out = 0
bench = 0
depth = 8
l2norm = 0
am = 0

# multi-threading screws up benchmarking
conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

class BlocksparseMatMulTest(tf.test.TestCase):

    def atestSparseProj(self):
        nhidden = 1024*8
        nproj   = 1024
        N       = 64

        with self.test_session(config=conf) as sess, tf.device("/gpu:0"):

            if one:
                X  = np.ones((nhidden,N), dtype=np.float32)
                Y  = np.ones((  nproj,N), dtype=np.float32)
                EX = np.ones((nhidden,N), dtype=np.float32)
                EY = np.ones((  nproj,N), dtype=np.float32)
            else:
                X  = np.random.uniform(-1.0, 1.0, (nhidden,N)).astype(np.float32)
                Y  = np.random.uniform(-1.0, 1.0, (  nproj,N)).astype(np.float32)
                EX = np.random.uniform(-1.0, 1.0, (nhidden,N)).astype(np.float32)
                EY = np.random.uniform(-1.0, 1.0, (  nproj,N)).astype(np.float32)

            x  = tf.constant(X)
            y  = tf.constant(Y)
            ex = tf.constant(EX)
            ey = tf.constant(EY)

            sproj = SparseProj(nhidden, nproj)
            lut   = sproj.gather_lut

            SLC = X[lut,:]
            ADD = X.copy()
            MUL = X.copy()
            ADD[lut,:] += Y
            MUL[lut,:] *= Y

            SLC_DX = np.zeros(x.shape)
            SLC_DX[lut,:] = EY

            ADD_DX = EX
            ADD_DY = EX[lut,:]

            MUL_DX = EX.copy()
            MUL_DX[lut,:] *= Y

            MUL_DY = EX[lut,:] * X[lut,:]

            slc_op = sproj.gather(x)
            mul_op = sproj.scatter_mul(x, y)
            add_op = sproj.scatter_add(x, y)

            slc = sess.run( slc_op )
            mul = sess.run( mul_op )
            add = sess.run( add_op ) # this op overwrites x, run last

            slc_dx,        = sess.run( tf.gradients(slc_op, [x  ], ey, aggregation_method=am) )
            add_dx, add_dy = sess.run( tf.gradients(add_op, [x,y], ex, aggregation_method=am) )
            mul_dx, mul_dy = sess.run( tf.gradients(mul_op, [x,y], ex, aggregation_method=am) ) # this op overwrites ex, run last

            for op, cpuA, devA in (
                ("slc:", SLC, slc),
                ("add:", ADD, add),
                ("mul:", MUL, mul),
                ("slc_dx:", SLC_DX, slc_dx),
                ("add_dx:", ADD_DX, add_dx),
                ("add_dy:", ADD_DY, add_dy),
                ("mul_dx:", MUL_DX, mul_dx),
                ("mul_dy:", MUL_DY, mul_dy),
            ):

                difA = abs(cpuA - devA)

                avgval  = np.average(abs(cpuA))
                maxdif  = difA.max()
                max_err = maxdif if avgval == 0 else maxdif / avgval

                l2_err = np.sqrt(np.square(difA).sum()) / np.sqrt(np.square(cpuA).sum())

                print("%s max_err%%:%10.8f L2_err: %12.10f" % (op, 100*max_err, l2_err))

                if out:
                    np.savetxt("out.txt",  difA, fmt='%5.1f')
                    np.savetxt("outC.txt", cpuA, fmt='%5.1f')
                    np.savetxt("outD.txt", devA, fmt='%5.1f')
                    exit()

    def atestBlocksparseMatMulCPU(self):
        # n, m = 64*8, 64
        # #layout = networkx.generators.barabasi_albert_graph(n, m)
        # layout = networkx.generators.random_graphs.watts_strogatz_graph(n, m*2, .2)
        # layout = networkx.adjacency_matrix(layout).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
        # layout[0:m,0:m] = 1

        # blocks = layout.sum()
        # print(100 * blocks / n**2)
        # print(layout.sum(axis=0).max())

        with self.test_session(config=conf) as sess, tf.device("/cpu:0"):
            for bsize, axis in ( (32,0), (16,0), (8,0) ): # (32,0), (16,0), (8,0)

                layout = np.ones((4*1024//bsize,4*1024//bsize), dtype=np.int32)

                bsmm = BlocksparseMatMul(layout, block_size=bsize, feature_axis=axis, name="test")

                if one:
                    W = np.ones(bsmm.w_shape,    dtype=np.float32)
                    X = np.ones(bsmm.i_shape(1), dtype=np.float32)
                else:
                    W = np.random.uniform(-1.0, 1.0, bsmm.w_shape   ).astype(np.float32)
                    X = np.random.uniform(-1.0, 1.0, bsmm.i_shape(1)).astype(np.float32)

                w = tf.constant(W)
                x = tf.constant(X)
                y = sess.run( bsmm(x, w, bench=bench) )

                #start = time()
                Y = bsmm.fprop_test(X, W)
                #print("np time:", round(time() - start, 2))

                difY = abs(Y - y)

                avgval  = np.average(abs(Y))
                maxdif  = difY.max()
                max_err = maxdif if avgval == 0 else maxdif / avgval

                l2_err = np.sqrt(np.square(difY).sum()) / np.sqrt(np.square(Y).sum())

                print("cpu max_err%%: %10.8f L2_err: %12.10f" % (100*max_err, l2_err))


    def testBlocksparseMatMul(self):

        # layout = np.zeros((2,2), dtype=np.int32)
        # layout[0,0] = 1

        n, m = 56*8, 8
        layout = networkx.generators.barabasi_albert_graph(n, m)
        #layout = networkx.generators.random_graphs.watts_strogatz_graph(n, m*2, .5)
        layout = networkx.adjacency_matrix(layout).toarray().astype(np.int32) + np.eye(n, dtype=np.int32)
        layout[0:m,0:m] = 1

        #layout[0:60,0:60] = 1
        #layout = np.ones((112,112), dtype=np.int32)
        #layout = np.ones((28*12,28*12), dtype=np.int32)
        # layout[0,0] = 1

        blocks = layout.sum()
        n = layout.shape[0]
        print(100 * blocks / n**2)
        print(layout.sum(axis=0).max())
        #exit()

        with self.test_session(config=conf) as sess, tf.device("/gpu:0"):

            for bsize, axis in ( (32,1), (32,0), (16,0), (8,0), ): # (32,0), (16,0), (8,0)

                bsmm = BlocksparseMatMul(layout, block_size=bsize, feature_axis=axis, name="test")

                if one:
                    W = np.ones(bsmm.w_shape, dtype=np.float32)
                    #W[:] += np.arange(8, dtype=np.float32).reshape(1,8)
                else:
                    W = np.random.uniform(-1.0, 1.0, bsmm.w_shape).astype(np.float32)


                # WW = np.zeros((bsmm.C, bsmm.K), dtype=np.float32)
                # for w, (c, k) in enumerate(bsmm.updat_list):
                #     WW[c*bsize:(c+1)*bsize, k*bsize:(k+1)*bsize] = W[w,:,:]

                w = tf.constant(W)

                # s1 = sess.run( bsmm.identity_init(gpu=True)(bsmm.w_shape) )
                # s2 = bsmm.identity_init(gpu=False)(bsmm.w_shape)
                # print("identity_init: ", (s1 - s2).max())

                for N in (64,): # 128,64,32,16,1,

                    if one:
                        X = np.ones(bsmm.i_shape(N), dtype=np.float32)
                        E = np.ones(bsmm.o_shape(N), dtype=np.float32)
                        #X[:] += np.arange(8, dtype=np.float32).reshape(8,1)
                    else:
                        X = np.random.uniform(-1.0, 1.0, bsmm.i_shape(N)).astype(np.float32)
                        E = np.random.uniform(-1.0, 1.0, bsmm.o_shape(N)).astype(np.float32)

                    x = tf.constant(X)
                    e = tf.constant(E)

                    for dtF, dtB in dtypes:

                        print("Axis:%d Bsize:%2d N:%d F:%s B:%s Params:%d" % (axis, bsize, N, dtF.name, dtB.name, bsize*bsize*blocks))

                        # compute in tensorflow
                        if l2norm:
                            w2 = bsmm.l2_normalize(w, dtype=dtF)
                        else:
                            w2 = ew.float_cast(w, dtype=dtF)

                        y = ew.float_cast(x, dtype=dtF)

                        for j in range(depth):
                            repeat = bench if bench and j==depth-1 else 0
                            y = bsmm(y, w2, dw_dtype=dtF, bench=repeat) # (bench and j==depth-1) (bench and j==0)

                        y = ew.float_cast(y, dtype=tf.float32, dx_dtype=dtB)
                        if bench: sess.run( y )
                        #y = sess.run( y )

                        d = tf.gradients(y, [x, w], e, aggregation_method=am)
                        if depth > 1:
                            d[1] = group_param_grads(d[1], 8)

                        y, (dx, dw) = sess.run( [y, d] )

                        if not bench:
                            # compute in numpy
                            if l2norm:
                                W2 = bsmm.l2_normalize_test(W)
                            else:
                                W2 = W

                            # YY = np.dot(WW.T, X)
                            # ZZ = np.dot(WW  , E)
                            # uu = np.dot( X  , E.T)
                            # UU = np.zeros(bsmm.w_shape, dtype=np.float32)
                            # for w, (c, k) in enumerate(bsmm.updat_list):
                            #     UU[w,:,:] = uu[c*bsize:(c+1)*bsize, k*bsize:(k+1)*bsize]


                            Ys = [X]
                            for j in range(depth):
                                Ys.append(bsmm.fprop_test(Ys[-1], W2))
                            Y = Ys.pop()

                            DW = np.zeros(bsmm.w_shape, dtype=np.float32)
                            DX = E
                            for j in range(depth):
                                DW += bsmm.updat_test(Ys.pop(), DX)
                                DX  = bsmm.bprop_test(DX, W2)
                            if l2norm:
                                DW = bsmm.l2_normalize_grad_test(W, DW)

                            for op, cpuA, devA in (
                                # ("YY:", YY,  y),
                                # ("ZZ:", ZZ, dx),
                                # ("UU:", UU, dw),
                                (" y:",  Y,  y),
                                ("dx:", DX, dx),
                                ("dw:", DW, dw),):

                                difA = abs(cpuA - devA)

                                avgval  = np.average(abs(cpuA))
                                maxdif  = difA.max()
                                max_err = maxdif if avgval == 0 else maxdif / avgval

                                l2_err = np.sqrt(np.square(difA).sum()) / np.sqrt(np.square(cpuA).sum())

                                #print("max_err: %5.3f, max_val: %7.3f, l1_err: %7.5f, l2_err: %7.5f" % (difO.max(), cpuO.max(), l1_err, l2_err))

                                print("%s max_err%%:%10.8f L2_err: %12.10f" % (op, 100*max_err, l2_err))

                                # rtol = 1e-4 if dtF is tf.float32 else 1e-1
                                # self.assertAllClose(devA, cpuA, rtol=rtol, atol=rtol)
                                if out:
                                    dim = bsmm.K if op == "dw:" else N
                                    np.savetxt("out.txt",  difA.reshape((-1,dim)), fmt='%5.1f')
                                    np.savetxt("outC.txt", cpuA.reshape((-1,dim)), fmt='%5.1f')
                                    np.savetxt("outD.txt", devA.reshape((-1,dim)), fmt='%5.1f')
                                    exit()
                            print("")

if __name__ == "__main__":
    print(sys.argv)
    tf.test.main(argv=["blocksparse_matmul_test.py","BlocksparseMatMulTest"])
