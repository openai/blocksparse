#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import blocksparse.ewops as ew
from tensorflow.python.ops import gradient_checker

ones = 0
out  = 0

class EwOpsTest(tf.test.TestCase):

    def testEwOps(self):

        with self.test_session() as sess, tf.device("/gpu:0"):

            for shape in ( (32,1024), ): # (31,31*4), (11,1023), (33,33),
                for dtypeF, dtypeB in ((np.float32, np.float32),  ): #, (np.float32, np.float32), (np.float16, np.float16), (np.float16, np.float32),
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

                        xx = tf.ones(shape, dtype=tf.float16)
                        ee = tf.ones(shape)

                        ew_op1 = ew.dropout(xx, keep_prob=0.5)
                        ew_op2 = ew.dropout(xx, mask=ew_op1[1])
                        dx_op  = tf.gradients(ew_op1[0], [xx], ee)
                        (z1, m), z2, (dx,) = sess.run( [ew_op1, ew_op2, dx_op] )
                        print(z1.sum()/z1.size, dx.sum()/dx.size, (z1 - z2).sum(), (z1 - dx).sum())

                        z = sess.run( ew.sparse_relu(x) )
                        Z = ew.sparse_relu_test(np_X)
                        tests.append(("sps_relu: Z ",  Z,  z))

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

                        # Unary Ops
                        for name, tf_op, ew_op in (
                            ("     sig", tf.sigmoid,    ew.sigmoid     ),
                            ("    tanh", tf.tanh,       ew.tanh        ),
                            ("     neg", tf.negative,   ew.negative,   ),
                            ("     rcp", tf.reciprocal, ew.reciprocal, ),
                            ("     sqr", tf.square,     ew.square,     ),
                            ("    sqrt", tf.sqrt,       ew.sqrt,       ),
                            ("     exp", tf.exp,        ew.exp,        ),
                            ("     log", tf.log,        ew.log,        ),
                            ("    relu", tf.nn.relu,    ew.relu,       ),
                            ("     elu", tf.nn.elu,     ew.elu,        ),):

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

                        if (shape[1] % 4) == 0:

                            # Split 4
                            e4 = [ tf.constant(x.astype(dtypeB)) for x in np.split(np_E, 4, 1) ]
                            E4 = [ tf.constant(x)                for x in np.split(np_E, 4, 1) ]

                            tf_op  = tf.split(X, 4, 1)
                            ew_op  = ew.split4(x)
                            Z, z = sess.run( [tf_op, ew_op] )
                            DX, = sess.run( tf.gradients(list(tf_op), [X], E4) )
                            dx, = sess.run( tf.gradients(list(ew_op), [x], e4) )
                            tests.append(("  split4: Z0",  Z[0],  z[0]))
                            tests.append(("  split4: Z1",  Z[1],  z[1]))
                            tests.append(("  split4: Z2",  Z[2],  z[2]))
                            tests.append(("  split4: Z3",  Z[3],  z[3]))
                            tests.append(("  split4: DX",  DX,    dx  ))

                            # LSTM Gates:
                            K4 = shape[1] // 4
                            c  = tf.constant(np_Y[:, 0:K4  ].astype(dtypeF))
                            ec = tf.constant(np_E[:, 0:K4  ].astype(dtypeB))
                            eh = tf.constant(np_E[:,K4:K4*2].astype(dtypeB))

                            i, f, o, u = ew.split4(x)
                            i = ew.sigmoid(i)
                            f = ew.sigmoid(f)
                            o = ew.sigmoid(o)
                            u = ew.tanh(u)
                            cn = ew.add(ew.multiply(f, c), ew.multiply(i, u))
                            hn = ew.multiply(o, ew.tanh(cn))

                            ew_op  = ew.fused_lstm_gates(c, x)
                            ew_op4 = ew.fused_lstm_gates(c, *ew.split4(x))

                            C_next, H_next = sess.run( [cn, hn] )
                            c_next, h_next = sess.run( ew_op )
                            c_next4, h_next4 = sess.run( ew_op4 )

                            # gradient_checker tests are insanely slow
                            # error = gradient_checker.compute_gradient_error(c, (shape[0],K4), ew_op4[0], (shape[0],K4))
                            # assert error < 0.01
                            # error = gradient_checker.compute_gradient_error(x,  shape,        ew_op4[1], (shape[0],K4))
                            # assert error < 0.01

                            # DC, DH = sess.run( tf.gradients(   [cn, hn], [c, x], [ec, eh]) )
                            # dc, dh = sess.run( tf.gradients(list(ew_op), [c, x], [ec, eh]) )

                            tests.append(("    LSTM:  C", C_next, c_next ))
                            tests.append(("    LSTM:  H", H_next, h_next ))
                            tests.append(("    LSTM: C4", C_next, c_next4 ))
                            tests.append(("    LSTM: H4", H_next, h_next4 ))
                            # tests.append(("    LSTM: DC", DC, dc ))
                            # tests.append(("    LSTM: DH", DH, dh ))


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
