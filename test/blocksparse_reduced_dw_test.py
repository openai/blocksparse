#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from blocksparse.matmul import blocksparse_reduced_dw

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

class BlocksparseReducedDWTest(tf.test.TestCase):

    def testBlocksparseReducedDW(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            ones     = 0
            norm     = 0
            accum    = 0
            blocks_x = 2
            blocks_y = 4
            bsize    = 32
            axis     = 0
            depth    = 8
            N        = 64
            scale    = 1.0 / (N * depth)
            shape_x  = [N, N]
            shape_y  = [N, N]
            shape_w  = (blocks_x, blocks_y)
            shape_x[axis] = bsize * blocks_x
            shape_y[axis] = bsize * blocks_y



            XS = list()
            YS = list()
            if ones:
                for i in range(depth):
                    XS.append(np.ones(shape_x, dtype=np.float32))
                    YS.append(np.ones(shape_y, dtype=np.float32))
                    if accum:
                        DWA = np.ones(shape_w, dtype=np.float32)

                    XS[0][:] += np.arange(64, dtype=np.float32).reshape(1,64)
            else:
                for i in range(depth):
                    XS.append(np.random.normal(0.0, 1.0, shape_x).astype(np.float16).astype(np.float32))
                    YS.append(np.random.normal(0.0, 1.0, shape_y).astype(np.float16).astype(np.float32))
                    if accum:
                        DWA = np.random.normal(0.0, 1.0, shape_w).astype(np.float32)

            feed_dict = dict()
            xs = list()
            ys = list()
            for i in range(depth):
                x = tf.placeholder(tf.float32, shape_x, name=f"x{i}")
                y = tf.placeholder(tf.float32, shape_y, name=f"y{i}")
                feed_dict[x] = XS[i]
                feed_dict[y] = YS[i]
                xs.append(bs.float_cast(x, dtype=tf.float16))
                ys.append(bs.float_cast(y, dtype=tf.float16))

            if accum:
                dwa = tf.placeholder(tf.float32, DWA.shape, name=f"dwa")
                feed_dict[dwa] = DWA
                #dwa = bs.float_cast(dwa, dtype=tf.float16)
                dw, x_red, y_red = blocksparse_reduced_dw(xs, ys, scale, [dwa], bsize=bsize, norm=norm, axis=axis)
            else:
                dw, x_red, y_red = blocksparse_reduced_dw(xs, ys, scale, [   ], bsize=bsize, norm=norm, axis=axis)

            #dw    = bs.float_cast(dw,    dtype=tf.float32)
            x_red = bs.float_cast(x_red, dtype=tf.float32)
            y_red = bs.float_cast(y_red, dtype=tf.float32)

            dw, x_red, y_red = sess.run([dw, x_red, y_red], feed_dict=feed_dict)

            if axis == 0:
                X_RED = np.zeros([blocks_x, depth, N], dtype=np.float32)
                Y_RED = np.zeros([blocks_y, depth, N], dtype=np.float32)

                for i in range(depth):
                    X = XS[i].reshape([blocks_x, bsize, N])
                    Y = YS[i].reshape([blocks_y, bsize, N])
                    if norm == 0:
                        X_RED[:,i,:] = np.max(np.abs(X), axis=1)
                        Y_RED[:,i,:] = np.max(np.abs(Y), axis=1)
                    else:
                        X_RED[:,i,:] = np.sqrt(np.sum(np.square(X), axis=1))
                        Y_RED[:,i,:] = np.sqrt(np.sum(np.square(Y), axis=1))

                DW = np.dot(X_RED.reshape(blocks_x, -1), Y_RED.reshape(blocks_y, -1).T) * scale

            else:
                X_RED = np.zeros([depth, N, blocks_x], dtype=np.float32)
                Y_RED = np.zeros([depth, N, blocks_y], dtype=np.float32)

                for i in range(depth):
                    X = XS[i].reshape([N, blocks_x, bsize])
                    Y = YS[i].reshape([N, blocks_y, bsize])
                    if norm == 0:
                        X_RED[i,:,:] = np.max(np.abs(X), axis=2)
                        Y_RED[i,:,:] = np.max(np.abs(Y), axis=2)
                    else:
                        X_RED[i,:,:] = np.sqrt(np.sum(np.square(X), axis=2))
                        Y_RED[i,:,:] = np.sqrt(np.sum(np.square(Y), axis=2))

                DW = np.dot(X_RED.reshape(-1, blocks_x).T, Y_RED.reshape(-1, blocks_y)) * scale

            if accum:
                DW += DWA

            print("BlocksparseReducedDW", norm, bsize, depth)
            for op, dev, cpu in [
                [ "xr", x_red, X_RED ],
                [ "yr", y_red, Y_RED ],
                [ "dw",    dw,    DW ],
            ]:
                #print(op, dev.shape, cpu.shape)
                self.compare_results(op, dev, cpu)


    def compare_results(self, op, dev, cpu):

        dif     = np.abs(cpu - dev)
        avgval  = np.average(abs(cpu))
        maxdif  = dif.max()
        max_err = maxdif if avgval == 0 else maxdif / avgval
        l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

        print("op:%3s, err:%17.12f, l2_err:%17.12f shape:%14s" % (op, maxdif, l2_err, str(cpu.shape)))

        if 0:
            np.savetxt("%s_dif.txt"%op, dif.reshape(-1, dif.shape[-1]), fmt='%6.3f')
            np.savetxt("%s_cpu.txt"%op, cpu.reshape(-1, cpu.shape[-1]), fmt='%6.3f')
            np.savetxt("%s_gpu.txt"%op, dev.reshape(-1, dev.shape[-1]), fmt='%6.3f')
            exit()


if __name__ == "__main__":
  tf.test.main()

# 2560*2560*1024*2 / (452.92 * 1000)
