#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
import blocksparse.transformer as trans
#from tensorflow.python.framework import function

ones   = 0
out    = 0
bench  = 0

batch = 4
heads = 4
ctx   = 16
state = 64*8
scale = 1.0 / np.sqrt(state)

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

# define inner block structure for masked softmax
def mask_callback(blk_shape, head_idx, qry_idx, key_idx, blk_idx):

    # default to enabled
    mask = np.ones(blk_shape, dtype=np.bool)

    # on the diagonal blocks mask out the upper diagonal
    if qry_idx == key_idx:
        for q, k in np.ndindex(blk_shape):
            if k > q:
                mask[q,k] = 0
        # if head == 0:
        #     print(mask.astype(np.uint8))
    return mask

class BlocksparseTransformerTest(tf.test.TestCase):

    def testBlocksparseTransformerDense(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for bsize in (8, 16, 32, 64):

                layout = np.ones([heads, ctx, ctx], dtype=np.bool)
                bst = trans.BlocksparseTransformer(layout, block_size=bsize)

                shape = (batch, ctx*bsize, heads*state)

                if ones:
                    cpuQ = np.ones(shape, dtype=np.float32)
                    cpuK = np.ones(shape, dtype=np.float32)
                    cpuV = np.ones(shape, dtype=np.float32)
                    cpuE = np.ones(shape, dtype=np.float32)
                else:
                    cpuQ = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuK = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuV = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)

                q = tf.placeholder(tf.float32,shape)
                k = tf.placeholder(tf.float32,shape)
                v = tf.placeholder(tf.float32,shape)
                e = tf.placeholder(tf.float32,shape)

                feed_dict = { q: cpuQ, k: cpuK, v: cpuV, e: cpuE }

                qf = ew.float_cast(q, dtype=tf.float16)
                kf = ew.float_cast(k, dtype=tf.float16)
                vf = ew.float_cast(v, dtype=tf.float16)

                w = bst.query_key_op(qf, kf)
                w = bst.softmax(w, scale=scale)
                y = bst.weight_value_op(w, vf)

                qf = trans.transpose_0213(tf.reshape(qf, [batch, ctx*bsize, heads, state]))
                kf = trans.transpose_0213(tf.reshape(kf, [batch, ctx*bsize, heads, state]))
                vf = trans.transpose_0213(tf.reshape(vf, [batch, ctx*bsize, heads, state]))
                W = tf.matmul(qf, kf, transpose_b=True)
                W = trans.softmax(W, scale=scale)
                Y = tf.matmul(W, vf)
                Y = tf.reshape(trans.transpose_0213(Y), [batch, ctx*bsize, heads*state])

                y = ew.float_cast(y, dtype=tf.float32)
                Y = ew.float_cast(Y, dtype=tf.float32)

                y, (dq, dk, dv) = sess.run( [ y, tf.gradients(y, [q, k, v], e) ], feed_dict )
                Y, (DQ, DK, DV) = sess.run( [ Y, tf.gradients(Y, [q, k, v], e) ], feed_dict )

                print("testBlocksparseTransformerDense", bsize)
                if not bench:
                    for op, dev, cpu in [
                        [ " Y",  y,  Y ],
                        [ "DV", dv, DV ],
                        [ "DK", dk, DK ],
                        [ "DQ", dq, DQ ],
                    ]:
                        self.compare_results(op, dev, cpu)

    def testBlocksparseTransformerSparse(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for bsize in (8, 16, 32, 64):

                layout = np.ones([heads, ctx, ctx], dtype=np.bool)
                for q, k in np.ndindex(ctx, ctx):
                    if k > q:
                        layout[:,q,k] = 0
                bst = trans.BlocksparseTransformer(layout, block_size=bsize, mask_callback=mask_callback)

                shape = (batch, ctx*bsize, heads*state)

                if ones:
                    cpuQ = np.ones(shape, dtype=np.float32)
                    cpuK = np.ones(shape, dtype=np.float32)
                    cpuV = np.ones(shape, dtype=np.float32)
                    cpuE = np.ones(shape, dtype=np.float32)
                else:
                    cpuQ = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuK = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuV = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)

                q = tf.placeholder(tf.float32, shape)
                k = tf.placeholder(tf.float32, shape)
                v = tf.placeholder(tf.float32, shape)
                e = tf.placeholder(tf.float32, shape)

                feed_dict = { q: cpuQ, k: cpuK, v: cpuV, e: cpuE }

                qf = ew.float_cast(q, dtype=tf.float16)
                kf = ew.float_cast(k, dtype=tf.float16)
                vf = ew.float_cast(v, dtype=tf.float16)

                w = bst.query_key_op(qf, kf)
                w = bst.masked_softmax(w, scale=scale)
                y = bst.weight_value_op(w, vf)

                y = ew.float_cast(y, dtype=tf.float32)

                dq, dk, dv = tf.gradients(y, [q, k, v], e)
                y, dq, dk, dv = sess.run( [ y, dq, dk, dv ], feed_dict )

                W = bst.nt_test(cpuQ, cpuK)
                W = bst.masked_softmax_test(W, scale=scale)
                Y = bst.nn_test(W, cpuV)

                DV = bst.tn_test(   W, cpuE)
                DW = bst.nt_test(cpuE, cpuV)

                DW = bst.masked_softmax_grad_test(DW, W, scale=scale)

                DQ = bst.nn_test(  DW, cpuK)
                DK = bst.tn_test(  DW, cpuQ)

                print("testBlocksparseTransformerSparse", bsize)
                if not bench:
                    for op, dev, cpu in [
                        [ " Y",  y,  Y ],
                        [ "DV", dv, DV ],
                        [ "DK", dk, DK ],
                        [ "DQ", dq, DQ ],
                    ]:
                        self.compare_results(op, dev, cpu)

    def testBlocksparseSoftmax(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for bsize in ( 8, 16, 32, 64, ): # 16, 32, 64

                # define outer block structure for blocksparse matmul
                layout = np.ones([1, ctx, ctx], dtype=np.bool)
                for q, k in np.ndindex(ctx, ctx):
                    if k > q:
                        layout[:,q,k] = 0
                #print(layout[0])

                bst = trans.BlocksparseTransformer(layout, heads=heads, block_size=bsize, mask_callback=mask_callback)

                shape = (batch, heads, bst.blocks, bsize, bsize)

                if ones:
                    cpuX = np.ones(shape, dtype=np.float32)
                    cpuE = np.ones(shape, dtype=np.float32)

                else:
                    cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)

                x = tf.placeholder(tf.float32, cpuX.shape)
                e = tf.placeholder(tf.float32, cpuE.shape)
                feed_dict = { x: cpuX, e: cpuE }

                xf = ew.float_cast(x, dtype=tf.bfloat16)

                y = bst.masked_softmax(xf, scale=scale)

                y = ew.float_cast(y, dtype=tf.float32)

                dx, = tf.gradients(y, [ x ], e)

                y, dx = sess.run( [ y, dx ], feed_dict )

                Y  = bst.masked_softmax_test(cpuX, scale=scale)
                DX = bst.masked_softmax_grad_test(cpuE, Y, scale=scale)

                print("testBlocksparseSoftmax", bsize)
                for op, dev, cpu in [
                    [ " Y",  y,  Y ],
                    [ "DX", dx, DX ],
                ]:
                    self.compare_results(op, dev, cpu)

    def testBlocksparseTransformerMatmul(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for bsize in (8, 16, 32, 64): # 8, 16, 32, 64

                layout = np.ones([1, ctx, ctx], dtype=np.bool)
                for q, k in np.ndindex(ctx, ctx):
                    if k > q:
                        layout[:,q,k] = 0
                #layout[:,0,:] = 1
                bst = trans.BlocksparseTransformer(layout, heads=heads, block_size=bsize)

                q_shape = (batch, ctx*bsize, heads*state)
                w_shape = (batch, heads, bst.blocks, bsize, bsize)

                if ones:
                    cpuQ = np.ones(q_shape, dtype=np.float32)
                    cpuK = np.ones(q_shape, dtype=np.float32)
                    cpuW = np.ones(w_shape, dtype=np.float32)
                    # cpuQ[0,0,0,:] = 1
                    # cpuK[0,0,0,:] = range(64)
                    # cpuW[0,0,0,0,:] = 1
                else:
                    cpuQ = np.random.uniform(-1.0, 1.0, q_shape).astype(np.float16).astype(np.float32)
                    cpuK = np.random.uniform(-1.0, 1.0, q_shape).astype(np.float16).astype(np.float32)
                    cpuW = np.random.uniform(-1.0, 1.0, w_shape).astype(np.float16).astype(np.float32)

                q = tf.placeholder(tf.float32, cpuQ.shape)
                k = tf.placeholder(tf.float32, cpuK.shape)
                w = tf.placeholder(tf.float32, cpuW.shape)

                feed_dict = { q: cpuQ, k: cpuK, w: cpuW }

                qf = ew.float_cast(q, dtype=tf.float16)
                kf = ew.float_cast(k, dtype=tf.float16)
                wf = ew.float_cast(w, dtype=tf.float16)

                nt = bst.nt_op(qf, kf, bench=bench)
                nn = bst.nn_op(wf, kf, bench=bench)
                tn = bst.tn_op(wf, qf, bench=bench)

                nt = ew.float_cast(nt, dtype=tf.float32)
                nn = ew.float_cast(nn, dtype=tf.float32)
                tn = ew.float_cast(tn, dtype=tf.float32)

                print("testBlocksparseTransformerMatmul", bsize)

                nt, nn, tn = sess.run( [ nt, nn, tn ], feed_dict ) # nt, nn, tn

                if not bench:

                    NT = bst.nt_test(cpuQ, cpuK)
                    NN = bst.nn_test(cpuW, cpuK)
                    TN = bst.tn_test(cpuW, cpuQ)

                    for op, dev, cpu in [
                        [ "NT", nt, NT ],
                        [ "NN", nn, NN ],
                        [ "TN", tn, TN ],
                    ]:
                        self.compare_results(op, dev, cpu)



    def compare_results(self, op, dev, cpu):

        dif     = np.abs(cpu - dev)
        avgval  = np.average(abs(cpu))
        maxdif  = dif.max()
        max_err = maxdif if avgval == 0 else maxdif / avgval
        l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())

        print("op:%3s, err:%17.12f, l2_err:%17.12f shape:%14s" % (op, maxdif, l2_err, str(cpu.shape)))

        if out:
            dim = cpu.shape[-1]
            np.savetxt("%s_dif.txt" % op, dif.reshape((-1,dim)), fmt='%.0f')
            np.savetxt("%s_cpu.txt" % op, cpu.reshape((-1,dim)), fmt='%.0f')
            np.savetxt("%s_dev.txt" % op, dev.reshape((-1,dim)), fmt='%.0f')
            #exit()


if __name__ == "__main__":
  tf.test.main()

# a = np.zeros((32,32), dtype=np.bool)
# for y, x in np.ndindex(a.shape):
#     if x <= y: a[y,x] = True
# b = np.packbits(a.reshape(-1,8)[:,::-1]).view(np.uint32)

# np.unpackbits(b.view(np.uint8))

# b = np.packbits(a.reshape(-1,8)[:,::-1])

