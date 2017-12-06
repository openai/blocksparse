#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.python.ops import gradient_checker

from blocksparse.conv  import BlocksparseConv, BlocksparseDeconv
from blocksparse.norms import batch_norm, batch_norm_inference, batch_norm_inf_test, batch_norm_test, batch_norm_grad_test

ones = 0
out  = 0
debug = 0


# Blocks can be any rectangular size.
# Blocks can be be uniform or non-uniform in size
# Blocks can overlap in C and/or K dim (or not)
# c and k values can be entirely random
B = 4
blockC = 32
blockK = 48
BCK_diagonal =  [
                    [
                        [b*blockC + c for c in range(blockC)],
                        [b*blockK + k for k in range(blockK)],
                    ] for b in range(B)
                ]
B = 8
overlapC = 8
overlapK = 16
blockC   = 16
blockK   = 32
BCK_overlap  =  [
                    [
                        [b*overlapC + c for c in range(blockC)],
                        [b*overlapK + k for k in range(blockK)],
                    ] for b in range(B)
                ]

configs = [
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(1,1,1), DHW=(1,1,32), dilates=(1,1,1), strides=(1,1,1), padding="VALID", edge_bias=False),
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(1,1,3), DHW=(1,1,32), dilates=(1,1,1), strides=(1,1,2), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(1,1,5), DHW=(1,1,32), dilates=(1,1,1), strides=(1,1,2), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseConv,   BCK=BCK_overlap,  TRS=(1,1,3), DHW=(1,1,32), dilates=(1,1,2), strides=(1,1,1), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(1,1,3), DHW=(1,1,32), dilates=(1,1,1), strides=(1,1,2), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(1,3,3), DHW=(1,8, 8), dilates=(1,1,1), strides=(1,1,1), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseConv,   BCK=BCK_overlap,  TRS=(1,3,3), DHW=(1,8, 8), dilates=(1,1,1), strides=(1,1,1), padding="VALID", edge_bias=False),
    dict(clss=BlocksparseConv,   BCK=BCK_diagonal, TRS=(3,3,3), DHW=(4,4, 4), dilates=(1,1,1), strides=(1,1,1), padding="SAME",  edge_bias=True),
    dict(clss=BlocksparseDeconv, BCK=BCK_diagonal, TRS=(1,1,3), DHW=(1,1,32), dilates=(1,1,1), strides=(1,1,2), padding="SAME",  edge_bias=True),
]
#def batch_norm_inf_test(x, g, b, m, v, epsilon=1e-12):
#def batch_norm_inference(x, g, b, m, v, epsilon=1e-12):

class BlocksparseConvTest(tf.test.TestCase):

    def testBlocksparseConv(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with self.test_session(config=config) as sess:
            with tf.device("/gpu:0"):

                count = 0
                for config in configs:
                    config["debug"] = debug
                    count += 1
                    name = "test" + str(count)
                    print("")
                    print(name)
                    with tf.name_scope(name):
                        clss = config.pop("clss")
                        bs_conv_op = clss(**config)

                        for dtypeF, dtypeB in ((np.float32, np.float32), ): #, (np.float16, np.float32)
                            dtypeF = np.dtype(dtypeF) # Forward + Weights
                            dtypeB = np.dtype(dtypeB) # Backwards

                            rtol = 1e-4 if dtypeF.type is np.float32 else 1e-1

                            with tf.name_scope("F%dB%d" % (dtypeF.itemsize, dtypeB.itemsize)):

                                K = bs_conv_op.o_shape(1)[1]
                                if ones:
                                    cpuF  = [ np.ones(bs_conv_op.f_shape(b), dtype=np.float32) for b in range(bs_conv_op.blocks) ]
                                    cpuEF = [ np.ones(bs_conv_op.f_shape(b), dtype=np.float32) for b in range(bs_conv_op.blocks) ]
                                    cpuG  = np.ones(K, dtype=np.float32)
                                    cpuB  = np.ones(K, dtype=np.float32)
                                else:
                                    cpuF  = [ np.random.uniform(-1.0, 1.0, bs_conv_op.f_shape(b)).astype(np.float32) for b in range(bs_conv_op.blocks) ]
                                    cpuEF = [ np.random.uniform(-1.0, 1.0, bs_conv_op.f_shape(b)).astype(np.float32) for b in range(bs_conv_op.blocks) ]
                                    cpuG  = np.random.uniform(-1.0, 1.0, (K,)).astype(np.float32)
                                    cpuB  = np.random.uniform(-1.0, 1.0, (K,)).astype(np.float32)

                                devF  = tf.constant(bs_conv_op.collapse_filter(cpuF,  dtypeF))
                                devEF = tf.constant(bs_conv_op.collapse_filter(cpuEF, dtypeB))
                                devG  = tf.constant(cpuG)
                                devB  = tf.constant(cpuB)

                                if bs_conv_op.edgeBiasDim:
                                    if ones:
                                        cpuEB = np.ones(bs_conv_op.edge_bias_shape(), dtype=np.float32)
                                    else:
                                        cpuEB = np.random.uniform(-1.0, 1.0, bs_conv_op.edge_bias_shape()).astype(np.float32)
                                    devEB = tf.constant(cpuEB)
                                else:
                                    cpuEB = devEB = None

                                for N in [1,2,28,]: #
                                    with tf.name_scope("N%d" % N):
                                        if ones:
                                            cpuI = np.ones(bs_conv_op.i_shape(N), dtype=np.float32)
                                            cpuE = np.ones(bs_conv_op.o_shape(N), dtype=np.float32)
                                            cpuA = np.ones(bs_conv_op.o_shape(N), dtype=np.float32)
                                        else:
                                            cpuI = np.random.uniform(-1.0, 1.0, bs_conv_op.i_shape(N)).astype(np.float32)
                                            cpuE = np.random.uniform(-1.0, 1.0, bs_conv_op.o_shape(N)).astype(np.float32)
                                            cpuA = np.random.uniform(-1.0, 1.0, bs_conv_op.o_shape(N)).astype(np.float32)
                                        devI = tf.constant(cpuI.astype(dtypeF))
                                        devE = tf.constant(cpuE.astype(dtypeB))
                                        devA = tf.constant(cpuA.astype(dtypeF))

                                    C = cpuI.shape[1]
                                    tests = list()

                                    # Conv and edge bias
                                    cpuO        = bs_conv_op.fprop_test(cpuF, cpuI, edge_bias=cpuEB)
                                    cpuZ        = bs_conv_op.bprop_test(cpuF, cpuE)
                                    cpuU, cpuDB = bs_conv_op.updat_test(cpuE, cpuI)

                                    op   = bs_conv_op(devF, devI, edge_bias=devEB)
                                    devO = sess.run( op )
                                    if cpuEB is None:
                                        devZ, devU        = sess.run( tf.gradients(op, [devI, devF       ], devE) )
                                    else:
                                        devZ, devU, devDB = sess.run( tf.gradients(op, [devI, devF, devEB], devE) )

                                    tests.append( ("conv fprop", devO, cpuO, N*K) )
                                    tests.append( ("conv bprop", devZ, cpuZ, N*C) )
                                    tests.append( ("conv updat", devU, cpuU, 1) )
                                    if cpuEB is not None:
                                        tests.append( ("edge dbias", devDB, cpuDB, K) )


                                    # L2 Norm without Gain
                                    if bs_conv_op.overlapK:
                                        cpuO    = bs_conv_op.l2_normalize_test(cpuF)
                                        cpuZ, _ = bs_conv_op.l2_normalize_grad_test(cpuF, cpuEF)

                                        op    = bs_conv_op.l2_normalize(devF, dtype=dtypeF)
                                        devO  = sess.run( op )
                                        devZ, = sess.run( tf.gradients(op, [devF], devEF) )
                                        tests.append( ("l2   fprop", devO,  cpuO,  1) )
                                        tests.append( ("l2   bprop", devZ,  cpuZ,  1) )
                                    # L2 Norm with Gain
                                    else:
                                        cpuO        = bs_conv_op.l2_normalize_test(cpuF, gain=cpuG)
                                        cpuZ, cpuDG = bs_conv_op.l2_normalize_grad_test(cpuF, cpuEF, gain=cpuG)

                                        op   = bs_conv_op.l2_normalize(devF, gain=devG, dtype=dtypeF)
                                        devO = sess.run( op )
                                        devZ, devDG = sess.run( tf.gradients(op, [devF, devG], devEF) )
                                        tests.append( ("l2g  fprop", devO,  cpuO,  1) )
                                        tests.append( ("l2g  bprop", devZ,  cpuZ,  1) )
                                        tests.append( ("l2g  dgain", devDG, cpuDG, K) ) #bs_conv_op.f_shape

                                        # error = gradient_checker.compute_gradient_error(devF, devF.get_shape().as_list(), op, devF.get_shape().as_list())
                                        # print(error)
                                        # assert error < 0.01
                                        # error = gradient_checker.compute_gradient_error(devG, devG.get_shape().as_list(), op, devF.get_shape().as_list())
                                        # print(error)
                                        # assert error < 0.01


                                    # batch norm test
                                    cpuO, cpuM, cpuV   = batch_norm_test(cpuA, cpuG, cpuB)
                                    cpuZ, cpuDG, cpuDB = batch_norm_grad_test(cpuE, cpuA, cpuG, cpuM, cpuV)

                                    bn_op = batch_norm(devA, devG, devB)
                                    devO, devM, devV   = sess.run( bn_op )
                                    devZ, devDG, devDB = sess.run( tf.gradients(bn_op[0], [devA, devG, devB], devE) )

                                    tests.append( ("bn   fprop", devO,  cpuO,  N*K) )
                                    tests.append( ("bn   mean ", devM,  cpuM,  K  ) )
                                    tests.append( ("bn   var  ", devV,  cpuV,  K  ) )
                                    tests.append( ("bn   bprop", devZ,  cpuZ,  N*K) )
                                    tests.append( ("bn   dgain", devDG, cpuDG, K  ) )
                                    tests.append( ("bn   dbias", devDB, cpuDB, K  ) )

                                    cpuO = batch_norm_inf_test (cpuA, cpuG, cpuB, cpuM, cpuV)
                                    op   = batch_norm_inference(devA, devG, devB, bn_op[1], bn_op[2])
                                    devO = sess.run( op )
                                    tests.append( ("bn   inf  ", devO,  cpuO,  N*K) )


                                    for op, dev, cpu, reshape in tests:

                                        if cpu is None:
                                            continue

                                        dev = np.array(dev)
                                        dif = cpu - dev

                                        avgval = abs(cpu).sum() / cpu.size
                                        maxdif = abs(dif).max()
                                        ratio  = maxdif / avgval

                                        print("dtypeF:f%d, dtypeB:f%d, N:%3d, op:%s avg:%17.12f maxdif:%17.12f ratio:%17.12f" % (dtypeF.itemsize, dtypeB.itemsize, N, op, avgval, maxdif, ratio))

                                        # print(dev[0,0,:,:])
                                        # print(cpu[0,0,:,:])
                                        # exit()

                                        if out:
                                            np.savetxt("out.txt",  dif.reshape(reshape, -1), fmt='%7.4f')
                                            np.savetxt("outC.txt", cpu.reshape(reshape, -1), fmt='%7.4f')
                                            np.savetxt("outD.txt", dev.reshape(reshape, -1), fmt='%7.4f')
                                            exit()

                                        self.assertAllClose(dev, cpu, rtol=rtol, atol=rtol)


if __name__ == "__main__":
  tf.test.main()
