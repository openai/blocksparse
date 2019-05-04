#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
import numpy       as np
import tensorflow  as tf
import blocksparse as bs

ones   = 0
out    = 0
bench  = 0

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

            batch = 2
            heads = 2
            state = 64*2
            scale = 1.0 / np.sqrt(state/heads)

            for bsize in (8, 16, 32, 64):

                ctxQ = 16
                ctxK = 16

                layout = np.ones([heads, ctxQ, ctxK], dtype=np.bool)
                bst = bs.BlocksparseTransformer(layout, block_size=bsize)

                shapeQ = (batch, ctxQ*bsize, heads*state)
                shapeK = (batch, ctxK*bsize, heads*state)

                if ones:
                    cpuQ = np.ones(shapeQ, dtype=np.float32)
                    cpuK = np.ones(shapeK, dtype=np.float32)
                    cpuV = np.ones(shapeK, dtype=np.float32)
                    cpuE = np.ones(shapeQ, dtype=np.float32)
                else:
                    cpuQ = np.random.uniform(-1.0, 1.0, shapeQ).astype(np.float16).astype(np.float32)
                    cpuK = np.random.uniform(-1.0, 1.0, shapeK).astype(np.float16).astype(np.float32)
                    cpuV = np.random.uniform(-1.0, 1.0, shapeK).astype(np.float16).astype(np.float32)
                    cpuE = np.random.uniform(-1.0, 1.0, shapeQ).astype(np.float16).astype(np.float32)

                q = tf.placeholder(tf.float32, shapeQ)
                k = tf.placeholder(tf.float32, shapeK)
                v = tf.placeholder(tf.float32, shapeK)
                e = tf.placeholder(tf.float32, shapeQ)

                feed_dict = { q: cpuQ, k: cpuK, v: cpuV, e: cpuE }

                qf = bs.float_cast(q, dtype=tf.float16)
                kf = bs.float_cast(k, dtype=tf.float16)
                vf = bs.float_cast(v, dtype=tf.float16)

                w = bst.query_key_op(qf, kf, bench=bench)
                w = bst.softmax(w, scale=scale)
                y = bst.weight_value_op(w, vf, bench=bench)

                qf = bs.transpose_0213(tf.reshape(qf, [batch, ctxQ*bsize, heads, state]))
                kf = bs.transpose_0213(tf.reshape(kf, [batch, ctxK*bsize, heads, state]))
                vf = bs.transpose_0213(tf.reshape(vf, [batch, ctxK*bsize, heads, state]))
                W = tf.matmul(qf, kf, transpose_b=True)
                W = bs.softmax(W, scale=scale)
                Y = tf.matmul(W, vf)
                Y = tf.reshape(bs.transpose_0213(Y), [batch, ctxQ*bsize, heads*state])

                y = bs.float_cast(y, dtype=tf.float32)
                Y = bs.float_cast(Y, dtype=tf.float32)

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

            batch = 2
            heads = 2
            ctx   = 16
            state = 64*2
            scale = 1.0 / np.sqrt(state/heads)
            dtype = tf.float32

            for bsize in ( 32, ): # 8, 16, 32, 64

                layout = np.ones([heads, ctx, ctx], dtype=np.bool)
                for q, k in np.ndindex(ctx, ctx):
                    if k > q:
                        layout[:,q,k] = 0
                bst = bs.BlocksparseTransformer(layout, block_size=bsize, mask_callback=mask_callback)

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

                qf = bs.float_cast(q, dtype=dtype)
                kf = bs.float_cast(k, dtype=dtype)
                vf = bs.float_cast(v, dtype=dtype)

                w = bst.query_key_op(qf, kf)
                a = bst.masked_softmax(w, scale=scale)
                y = bst.weight_value_op(a, vf)

                w = bs.float_cast(w, dtype=tf.float32)
                a = bs.float_cast(a, dtype=tf.float32)
                y = bs.float_cast(y, dtype=tf.float32)

                dq, dk, dv = tf.gradients(y, [q, k, v], e)
                w, a, y, dq, dk, dv = sess.run( [ w, a, y, dq, dk, dv ], feed_dict )

                W = bst.nt_test(cpuQ, cpuK)
                A = bst.masked_softmax_test(W, scale=scale)
                Y = bst.nn_test(A, cpuV)

                DV = bst.tn_test(   A, cpuE)
                DW = bst.nt_test(cpuE, cpuV)

                DW = bst.masked_softmax_grad_test(DW, A, scale=scale)

                DQ = bst.nn_test(  DW, cpuK)
                DK = bst.tn_test(  DW, cpuQ)

                print("testBlocksparseTransformerSparse", 32)
                if not bench:
                    for op, dev, cpu in [
                        [  "W",  w,  W ],
                        [  "A",  a,  A ],
                        [  "Y",  y,  Y ],
                        [ "DV", dv, DV ],
                        [ "DK", dk, DK ],
                        [ "DQ", dq, DQ ],
                    ]:
                        self.compare_results(op, dev, cpu)

    def testBlocksparseTransformerMatmul(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for bsize in ( 32, ): # 8, 16, 32, 64

                dtype_qk = tf.float32
                dtype_w  = tf.bfloat16
                ones  = 0
                bench = 0
                batch = 2
                heads = 4
                ctx   = 16
                state = 64*2
                scale = 1.0 # / np.sqrt(state/heads)

                ctxQ = ctx
                ctxK = ctx # *2

                layout = np.ones([1, ctxQ, ctxK], dtype=np.bool)
                for q, k in np.ndindex(ctx, ctx):
                    if k > q:
                        layout[:,q,k] = 0
                #layout[:,0,:] = 1
                bst = bs.BlocksparseTransformer(layout, heads=heads, block_size=bsize, mask_callback=mask_callback)

                q_shape = (batch, ctxQ*bsize, heads*state)
                k_shape = (batch, ctxK*bsize, heads*state)
                w_shape = (batch, heads, bst.blocks, bsize, bsize)

                if ones:
                    cpuQ = np.ones(q_shape, dtype=np.float32)
                    cpuK = np.ones(k_shape, dtype=np.float32)
                    cpuW = np.ones(w_shape, dtype=np.float32)
                    # cpuQ[0,:,:] = np.eye(bsize, dtype=np.float32)
                    # cpuK[0,:,:] = np.eye(bsize, dtype=np.float32)
                    # cpuW[0,0,0,:,:] = np.eye(bsize, dtype=np.float32)
                    # cpuQ[0,0,0,:] = 1
                    # cpuK[0,0,0,:] = range(64)
                    # cpuW[0,0,0,0,:] = 1
                else:
                    cpuQ = np.random.uniform(-1.0, 1.0, q_shape).astype(np.float16).astype(np.float32)
                    cpuK = np.random.uniform(-1.0, 1.0, k_shape).astype(np.float16).astype(np.float32)
                    cpuW = np.random.uniform(-1.0, 1.0, w_shape).astype(np.float16).astype(np.float32)

                q = tf.placeholder(tf.float32, cpuQ.shape)
                k = tf.placeholder(tf.float32, cpuK.shape)
                w = tf.placeholder(tf.float32, cpuW.shape)

                feed_dict = { q: cpuQ, k: cpuK, w: cpuW }

                qf = bs.float_cast(q, dtype=dtype_qk)
                kf = bs.float_cast(k, dtype=dtype_qk)
                wf = bs.float_cast(w, dtype=dtype_w)

                nt = bst.nt_op(qf, kf, bench=bench)
                nn = bst.nn_op(wf, kf, bench=bench)
                tn = bst.tn_op(wf, qf, bench=bench)

                nt = bs.float_cast(nt, dtype=tf.float32)
                nn = bs.float_cast(nn, dtype=tf.float32)
                tn = bs.float_cast(tn, dtype=tf.float32)

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

    def atestBlocksparseSoftmax(self):

        batch = 1
        heads = 1
        key   = 7

        def checker_callback(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
            mask = np.ones(blk_shape, dtype=np.bool)
            mask[::2,1::2] = False
            mask[1::2,::2] = False
            return mask

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            # for ctx in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096): #16, 32, 64, 128, 256, 512, 1024, 2048, 4096
            #     for bsize in (8, 16, 32, 64,): # 8, 16, 32, 64,
            #         if bsize * (ctx+0) <= 32768:

            for ctx in (16,): #16, 32, 64, 128, 256, 512, 1024, 2048, 4096
                for bsize in (8, 16, 32, 64, ): # 8, 16, 32, 64,
                    if bsize * (ctx) <= 32768:

                        # define outer block structure for blocksparse matmul
                        layout = np.ones([heads, ctx, ctx], dtype=np.bool)

                        bst = bs.BlocksparseTransformer(layout, heads=heads, block_size=bsize, mask_callback=checker_callback) # checker_callback

                        shape = (batch, heads, bst.blocks, bsize, bsize)
                        print(shape)

                        if ones:
                            cpuX = np.ones(shape, dtype=np.float32)
                            cpuE = np.ones(shape, dtype=np.float32)

                        else:
                            cpuX = np.random.normal(0.0, 1.0, shape).astype(np.float16).astype(np.float32)
                            cpuE = np.random.normal(0.0, 1.0, shape).astype(np.float16).astype(np.float32)

                        # np.savetxt("cpuX.txt", cpuX.reshape((-1,bsize)), fmt='%5.2f')

                        # for i, a in enumerate(np.max(cpuX.reshape(-1,bsize), axis=1)):
                        #     print("%2d %.2f" % (i, a))
                        # print()

                        x = tf.placeholder(tf.float32, cpuX.shape)
                        e = tf.placeholder(tf.float32, cpuE.shape)
                        feed_dict = { x: cpuX, e: cpuE }

                        xf = bs.float_cast(x, dtype=tf.bfloat16)

                        y = bst.masked_softmax(xf, scale=0.5, autoregress_at_key=key)

                        y = bs.float_cast(y, dtype=tf.float32)

                        dx, = tf.gradients(y, [ x ], e)

                        y, dx = sess.run( [ y, dx ], feed_dict )

                        Y  = bst.masked_softmax_test(cpuX, scale=0.5, autoregress_at_key=key)
                        DX = bst.masked_softmax_grad_test(cpuE, Y, scale=0.5)

                        print("testBlocksparseSoftmax", ctx*bsize, bsize)
                        for op, dev, cpu in [
                            [  "Y",  y,  Y ],
                            [ "DX", dx, DX ],
                        ]:
                            self.compare_results(op, dev, cpu)

    def testSoftmaxCrossEntropy(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            N = 3 # 80 * 16
            for K in (10, 256, 512, 1024*8, 1024*16, 1024*32, 1024*64,): #10, 256, 512, 1024*8, 1024*16, 1024*32, 1024*64

                np.random.seed(int(time()))
                #cpuX = np.random.uniform(-20.0, 20.0, (N, K)).astype(np.float16).astype(np.float32) #65504
                cpuX = np.random.normal(0.0, 1.0, (N, K)).astype(np.float16).astype(np.float32)
                cpuE = np.random.normal(0.0, 1.0, (N,  )).astype(np.float16).astype(np.float32)
                cpuI = np.random.randint(0, K, size=(N,  ),  dtype=np.uint16)

                x = tf.placeholder(tf.float32, cpuX.shape)
                e = tf.placeholder(tf.float32, cpuE.shape)
                i = tf.placeholder(tf.uint16,  cpuI.shape)
                feed_dict = { x: cpuX, i: cpuI, e: cpuE }

                xf = bs.float_cast(x, dtype=tf.float16)
                y = bs.softmax_cross_entropy(logits=xf, labels=i)

                Y = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=tf.cast(i, tf.int32))

                y, (dx,) = sess.run( [ y, tf.gradients(y, [x], e) ], feed_dict )
                Y, (DX,) = sess.run( [ Y, tf.gradients(Y, [x], e) ], feed_dict )

                print("testSoftmaxCrossEntropy", K)

                if not bench:
                    for op, dev, cpu in [
                        [  "Y",  y,  Y ],
                        [ "DX", dx, DX ],
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
            np.savetxt("%s_dif.txt" % op, dif.reshape((-1,dim)), fmt='%2.0f') #7.5 5.3
            np.savetxt("%s_cpu.txt" % op, cpu.reshape((-1,dim)), fmt='%2.0f') #7.5 5.3
            np.savetxt("%s_dev.txt" % op, dev.reshape((-1,dim)), fmt='%2.0f') #7.5 5.3
            exit()


if __name__ == "__main__":
  tf.test.main()

# a = np.zeros((32,32), dtype=np.bool)
# for y, x in np.ndindex(a.shape):
#     if x <= y: a[y,x] = True
# b = np.packbits(a.reshape(-1,8)[:,::-1]).view(np.uint32)

# np.unpackbits(b.view(np.uint8))

# b = np.packbits(a.reshape(-1,8)[:,::-1])

