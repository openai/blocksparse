#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
import blocksparse.transformer as trans
from tensorflow.python.ops import gradient_checker

out    = 0
bench  = 0
shapes = [
    ( 2,  2, 1024, 1024),
    ( 4,  4,  768,  768),
    ( 4,  4,  544,  544),
    ( 4,  4,  512,  512),
    ( 8,  8,  256,  256),
    (16, 16,  128,  128),
    (32, 32,   64,   64),
    (64, 64,   32,   32),
    # (1, 2, 1024, 1024),
    # (1, 2,  512,  512),
    # (1, 2,  256,  256),
    # (1, 2,  128,  128),
    # (1, 2,   64,   64),
    # (1, 2,   32,   32),
    # (1, 2, 1024, 1024-1),
    # (1, 2,  512,  512+1),
    # (1, 2,  256,  256+1),
    # (1, 2,  128,  128+1),
    # (1, 2,   64,   64+1),
    # (1, 2,   32,   32+1),
]

class TopKTest(tf.test.TestCase):

    def testTopK(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            for shape in shapes:

                topK = shape[-1] // 4 # 25% sparsity

                np.random.seed(int(time()))
                cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                cpuE = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)

                X = tf.placeholder(tf.float32, cpuX.shape)
                E = tf.placeholder(tf.float32, cpuE.shape)

                for mask_dims in (0, 2, 3):

                    if mask_dims == 0:
                        mask = M = m_shape = None
                        feed_dict = { X: cpuX, E: cpuE }

                    else:
                        m_shape = [1 for n in shape]
                        m_shape[-mask_dims:] = shape[-mask_dims:]
                        mask = np.zeros(m_shape, dtype=np.float32)

                        if mask_dims == 2:
                            for y, x in np.ndindex(mask.shape[-2:]):
                                if x <= y: mask[:,:,y,x] = 3.0
                        elif mask_dims == 3:
                            for z, y, x in np.ndindex(mask.shape[-3:]):
                                if x <= y: mask[:,z,y,x] = (z+1)*3.0

                        M = tf.placeholder(tf.float32, mask.shape)
                        feed_dict = { X: cpuX, E: cpuE, M: mask }

                    for dtype in (tf.float32, ):  #tf.float16, tf.bfloat16

                        rtol = 1e-4 if dtype is tf.float32 else 1e-1

                        Y = ew.float_cast(X, dtype=dtype)

                        #Y = trans.masked_top_k_softmax(Y, topK, mask=M, scale=2.0)

                        Y = trans.masked_softmax(Y, mask=M, scale=2.0, bench=bench)

                        Y = ew.float_cast(Y, dtype=tf.float32, dx_dtype=dtype)
                        D = tf.gradients(Y, [X], E)

                        #devY, = sess.run( [Y], feed_dict)
                        devY, (devDX,) = sess.run( [Y, D], feed_dict)
                        #devY, (devDX,), tfY = sess.run( [Y, D, tf.nn.top_k(X, topK)], feed_dict)

                        # gradient_checker tests are insanely slow
                        # if True:
                        #     x = tf.constant(cpuX)
                        #     m = tf.constant(mask)
                        #     y = trans.masked_top_k_softmax(x, topK, mask=m)

                        # error = gradient_checker.compute_gradient_error(x, shape, y, shape) #, extra_feed_dict={ x: cpuX, m: mask }
                        # assert error < 0.01, error

                        if bench == 0:

                            # cpuY  = trans.masked_top_k_softmax_test(cpuX, topK, mask=mask, scale=2.0)
                            # cpuDX = trans.masked_softmax_grad_test(cpuE, cpuY, mask=mask, scale=2.0)

                            cpuY  = trans.masked_softmax_test(cpuX, mask=mask, scale=2.0)
                            cpuDX = trans.masked_softmax_grad_test(cpuE, cpuY, mask=mask, scale=2.0)
                            difY  = np.abs(cpuY -  devY)
                            difDX = np.abs(cpuDX - devDX)
                            cntY  = (difY  > rtol).astype(np.int).sum() / difY.size
                            cntDX = (difDX > rtol).astype(np.int).sum() / difDX.size

                            print("%s, shape:%18s, mask:%18s, errY:%.5f, errDX:%.5f" % (dtype.name, str(shape), str(m_shape), cntY, cntDX))

                            if out:
                                np.savetxt( "cpuY.txt",  cpuY.reshape(-1,shape[-1]), fmt="%6.3f")
                                np.savetxt( "devY.txt",  devY.reshape(-1,shape[-1]), fmt="%6.3f")
                                np.savetxt("cpuDX.txt", cpuDX.reshape(-1,shape[-1]), fmt="%6.3f")
                                np.savetxt("devDX.txt", devDX.reshape(-1,shape[-1]), fmt="%6.3f")
                                np.savetxt("difDX.txt", difDX.reshape(-1,shape[-1]), fmt="%6.3f")

if __name__ == "__main__":

    tf.test.main()
