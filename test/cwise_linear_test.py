#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from operator import mul

if sys.version_info >= (3, 0):
    from functools import reduce

from blocksparse.conv import cwise_linear, cwise_linear_test, cwise_linear_grad_test

ones = 0
out  = 0

class BlocksparseConvTest(tf.test.TestCase):

    def testBlocksparseConv(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with self.test_session(config=config) as sess:
            with tf.device("/gpu:0"):

                for i, shape in enumerate( ((1,32,32),(64,64,32),(8,64,4,4),(8,64,16,16),(8,64,32,32),(8,64,8,8,8),) ):

                    DHW = reduce(mul, shape[2:], 1)

                    for dtypeF, dtypeB in ((np.float32, np.float32), (np.float16, np.float16),  ): #, (np.float16, np.float16), (np.float16, np.float32),
                        dtypeF = np.dtype(dtypeF) # Forward + Normalized Weights
                        dtypeB = np.dtype(dtypeB) # Backwards

                        rtol = 1e-4 if dtypeF.type is np.float32 else 1e-1

                        with tf.variable_scope("F%dB%d" % (dtypeF.itemsize, dtypeB.itemsize)):

                            if ones:
                                cpuX = np.ones(shape)
                                cpuE = np.ones(shape)
                                cpuA = np.ones(shape[1])
                                cpuB = np.ones(shape[1])
                            else:
                                cpuX = np.random.uniform(-1.0, 1.0, shape)
                                cpuE = np.random.uniform(-1.0, 1.0, shape)
                                cpuA = np.random.uniform(-1.0, 1.0, shape[1])
                                cpuB = np.random.uniform(-1.0, 1.0, shape[1])

                            devX = tf.constant(cpuX.astype(dtypeF))
                            devE = tf.constant(cpuE.astype(dtypeB))
                            devA = tf.constant(cpuA.astype(np.float32))
                            devB = tf.constant(cpuB.astype(np.float32))

                            # compute in numpy
                            cpuY0 = cwise_linear_test(cpuX, a=cpuA, b=cpuB)
                            cpuDX0, cpuDA0, cpuDB0 = cwise_linear_grad_test(cpuE, cpuX, a=cpuA)

                            cpuY1 = cwise_linear_test(cpuX, a=cpuA)
                            cpuDX1, cpuDA1, _ = cwise_linear_grad_test(cpuE, cpuX, a=cpuA)

                            cpuY2 = cwise_linear_test(cpuX, b=cpuB)
                            cpuDX2, _, cpuDB2 = cwise_linear_grad_test(cpuE, cpuX)

                            # compute in tensorflow
                            fprop0 = cwise_linear(devX, a=devA, b=devB)
                            devY0 = sess.run( fprop0 )
                            devDX0, devDA0, devDB0 = sess.run( tf.gradients(fprop0, [devX, devA, devB], devE) )

                            fprop1 = cwise_linear(devX, a=devA)
                            devY1 = sess.run( fprop1 )
                            devDX1, devDA1 = sess.run( tf.gradients(fprop1, [devX, devA], devE) )

                            fprop2 = cwise_linear(devX, b=devB)
                            devY2 = sess.run( fprop2 )
                            devDX2, devDB2 = sess.run( tf.gradients(fprop2, [devX, devB], devE) )

                            for op, devT, cpuT, reshape in (
                                (" devY0", devY0,  cpuY0,  (-1, DHW)),
                                ("devDX0", devDX0, cpuDX0, (-1, DHW)),
                                ("devDA0", devDA0, cpuDA0, shape[1]),
                                ("devDB0", devDB0, cpuDB0, shape[1]),
                                (" devY1", devY1,  cpuY1,  (-1, DHW)),
                                ("devDX1", devDX1, cpuDX1, (-1, DHW)),
                                ("devDA1", devDA1, cpuDA1, shape[1]),
                                (" devY2", devY2,  cpuY2,  (-1, DHW)),
                                ("devDX2", devDX2, cpuDX2, (-1, DHW)),
                                ("devDB2", devDB2, cpuDB2, shape[1]),):

                                devT = np.array(devT)
                                difA = cpuT - devT

                                avgval = abs(cpuT).sum() / cpuT.size
                                maxdif = abs(difA).max()
                                ratio  = maxdif / avgval

                                print("dtypeF:f%d, dtypeB:f%d, shape:%s, op:%s err:%17.12f" % (dtypeF.itemsize, dtypeB.itemsize, str(shape), op, ratio))

                                # print(devT[0,0,:,:])
                                # print(cpuT[0,0,:,:])
                                # exit()

                                if out:
                                    np.savetxt("out.txt",  difA.reshape(reshape), fmt='%5.2f')
                                    np.savetxt("outC.txt", cpuT.reshape(reshape), fmt='%5.2f')
                                    np.savetxt("outD.txt", devT.reshape(reshape), fmt='%5.2f')
                                    exit()

                                self.assertAllClose(devT, cpuT, rtol=rtol, atol=rtol)


if __name__ == "__main__":
  tf.test.main()
