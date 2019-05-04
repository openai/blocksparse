#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
import math
#from tensorflow.python.ops import gradient_checker

ones = 0
out  = 0

def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3.0))))

def swish(x):
    return x * tf.nn.sigmoid(x)

def fast_gelu(x):
    return x * tf.nn.sigmoid(1.702 * x)

class EwOpsTest(tf.test.TestCase):

    def testEwOps(self):

        with self.test_session() as sess, tf.device("/gpu:0"):

            for shape in ( (32,1024), ): # (31,31*4), (11,1023), (33,33),
                for dtypeF, dtypeB in ((np.float16, np.float16), (np.float32, np.float32) ): #, (np.float32, np.float32), (np.float16, np.float16), (np.float16, np.float32),
                    dtypeF = np.dtype(dtypeF) # Forward
                    dtypeB = np.dtype(dtypeB) # Backwards

                    rtol = 1e-4 if dtypeF.type is np.float32 else 1e-1

                    with tf.name_scope("S%dx%dF%dB%d" % (shape[0], shape[1], dtypeF.itemsize, dtypeB.itemsize)):

                        if ones:
                            np_X = np.ones(shape, dtype=np.float32)
                            np_Y = np.ones(shape, dtype=np.float32)
                            np_E = np.ones(shape, dtype=np.float32)
                            np_B = np.ones((1,shape[1]), dtype=np.float32)
                        else:
                            # np_X = np.random.normal(0.0, 10.0, shape).astype(dtypeF).astype(np.float32)
                            # np_E = np.random.normal(0.0, 10.0, shape).astype(dtypeF).astype(np.float32)
                            # np_X.fill(10.0)

                            np_X = np.random.uniform(0.01, 1.0, shape).astype(dtypeF).astype(np.float32)
                            np_Y = np.random.uniform(0.01, 1.0, shape).astype(dtypeF).astype(np.float32)
                            np_E = np.random.uniform(0.01, 1.0, shape).astype(dtypeB).astype(np.float32)
                            np_B = np.random.uniform(0.01, 1.0, (1,shape[1])).astype(np.float32)

                        x = tf.constant(np_X.astype(dtypeF))
                        y = tf.constant(np_Y.astype(dtypeF))
                        e = tf.constant(np_E.astype(dtypeB))
                        b = tf.constant(np_B)

                        X = tf.constant(np_X)
                        Y = tf.constant(np_Y)
                        E = tf.constant(np_E)
                        B = tf.constant(np_B)

                        tests = list()

                        # xx = tf.ones(shape, dtype=tf.float32)
                        # ee = tf.ones(shape, dtype=tf.float32)
                        # ew_op1 = ew.dropout(xx, keep_prob=0.5,  scale=2.0)
                        # ew_op2 = ew.dropout(xx, mask=ew_op1[1], scale=2.0)
                        # dx_op  = tf.gradients(ew_op1[0], [xx], ee)
                        # (z1, m), z2, (dx,) = sess.run( [ew_op1, ew_op2, dx_op] )
                        # #print(dx[0,0:8])
                        # print(z1.sum()/z1.size, dx.sum()/dx.size, (z1 - z2).sum(), (z1 - dx).sum())

                        # z = sess.run( ew.sparse_relu(x) )
                        # Z = ew.sparse_relu_test(np_X)
                        # tests.append(("sps_relu: Z ",  Z,  z))

                        # Non-Broadcast Binary Ops
                        for name, tf_op, ew_op in (
                            ("     add", tf.add,      ew.add      ),
                            ("     mul", tf.multiply, ew.multiply ),
                            ("     sub", tf.subtract, ew.subtract ),
                            ("     div", tf.divide,   ew.divide   ),
                            ("     max", tf.maximum,  ew.maximum  ),
                            ("     min", tf.minimum,  ew.minimum  ),):

                            # I think tf doesn't use fmaxf/fminf and hence has different behaviour for equal numbers.
                            # In fp32 the chance for equality is very small, but not so in fp16
                            if name[-3:] in ("max","min") and  dtypeF.type is np.float16:
                                continue

                            tf_op  = tf_op(X,Y)
                            ew_op  = ew_op(x,y)
                            Z, z   = sess.run( [tf_op, ew_op] )
                            DX, DY = sess.run( tf.gradients(tf_op, [X,Y], E) )
                            dx, dy = sess.run( tf.gradients(ew_op, [x,y], e) )
                            tests.append((name+": Z ",  Z,  z))
                            tests.append((name+": DX", DX, dx))
                            tests.append((name+": DY", DY, dy))

                        for name, tf_op, ew_op in (
                            ("   add_n", tf.add_n,    ew.add_n8_op),):

                            tf_op2  = tf_op([X,Y])
                            ew_op2  = ew_op([x,y])
                            tf_op3  = tf_op([X,Y,E])
                            ew_op3  = ew_op([x,y,e])
                            Z2, z2   = sess.run( [tf_op2, ew_op2] )
                            Z3, z3   = sess.run( [tf_op3, ew_op3] )
                            tests.append((name+": Z2",  Z2,  z2))
                            tests.append((name+": Z3",  Z3,  z3))

                        # Unary Ops
                        for name, tf_op, ew_op in (
                            ("      sig", tf.sigmoid,    ew.sigmoid     ),
                            ("     tanh", tf.tanh,       ew.tanh        ),
                            ("      neg", tf.negative,   ew.negative,   ),
                            ("      rcp", tf.reciprocal, ew.reciprocal, ),
                            ("      sqr", tf.square,     ew.square,     ),
                            ("     sqrt", tf.sqrt,       ew.sqrt,       ),
                            ("      exp", tf.exp,        ew.exp,        ),
                            ("      log", tf.log,        ew.log,        ),
                            ("     relu", tf.nn.relu,    ew.relu,       ),
                            ("      elu", tf.nn.elu,     ew.elu,        ),
                            ("     gelu", gelu,          ew.gelu,       ),
                            ("    swish", swish,         ew.swish,      ),
                            ("fast_gelu", fast_gelu,     ew.fast_gelu,  ),):

                            tf_op = tf_op(X)
                            ew_op = ew_op(x)
                            Z, z = sess.run( [tf_op, ew_op] )
                            DX, = sess.run( tf.gradients(tf_op, [X], E) )
                            dx, = sess.run( tf.gradients(ew_op, [x], e) )
                            tests.append((name+": Z ",  Z,  z))
                            tests.append((name+": DX", DX, dx))

                        # Broadcast Binary Ops
                        for name, tf_op, ew_op in (
                            ("bias_add", tf.add,      ew.add,    ),
                            ("bias_mul", tf.multiply, ew.multiply),):

                            tf_op  = tf_op(X,B)
                            ew_op  = ew_op(x,b)
                            Z, z   = sess.run( [tf_op, ew_op] )
                            DX, DB = sess.run( tf.gradients(tf_op, [X,B], E) )
                            dx, db = sess.run( tf.gradients(ew_op, [x,b], e) )
                            tests.append((name+": Z ",  Z,  z))
                            tests.append((name+": DX", DX, dx))
                            tests.append((name+": DB", DB, db))


                        # Up Cast
                        ew_op = ew.float_cast(x, dtype=tf.float32, dx_dtype=dtypeB.type)
                        z     = sess.run( ew_op )
                        dx,   = sess.run( tf.gradients(ew_op, [x], e) )
                        tests.append(("  upCast: Z ", np_X,  z))
                        tests.append(("  upCast: DX", np_E, dx))


                        #Down Cast
                        if dtypeF.type is np.float32:
                            Z   = np_X.astype(np.float16)
                            DX  = np_E.astype(np.float16)
                            e16 = tf.constant(DX)
                            ew_op = ew.float_cast(x, dtype=tf.float16)
                            z     = sess.run( ew_op )
                            dx,   = sess.run( tf.gradients(ew_op, [x], e16) )
                            tests.append(("downCast: Z ", Z,  z ))
                            tests.append(("downCast: DX", DX, dx))

                        for op, tfT, ewT in (tests):

                            dif = tfT - ewT

                            avgval = abs(tfT).sum() / tfT.size
                            maxdif = abs(dif).max()
                            ratio  = maxdif / avgval

                            print("dtypeF:f%d, dtypeB:f%d, shape:%s, op:%s err:%17.12f" % (dtypeF.itemsize, dtypeB.itemsize, str(shape), op, ratio))

                            # print(ewT[0,0,:,:])
                            # print(tfT[0,0,:,:])
                            # exit()

                            if out: # and ratio > 1.0:
                                np.savetxt("out.txt",  dif, fmt='%5.2f')
                                np.savetxt("outC.txt", tfT, fmt='%5.2f')
                                np.savetxt("outD.txt", ewT, fmt='%5.2f')
                                exit()

                            #self.assertAllClose(cpuT, ewT, rtol=rtol, atol=rtol)


if __name__ == "__main__":
  tf.test.main()
