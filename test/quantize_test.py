#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from struct import pack, unpack
from time import time

class QuantizeTest(tf.test.TestCase):

    def testQuantize(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        np.random.seed(int(time()))

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

            # max of 80 SMs on the GPU
            # 3 lsfr's and a max of 1024 threads active
            # Note tf does not supporet int32 variables!
            entropy_init = np.random.randint(-(1<<31), (1<<31), size=80*3*1024, dtype=np.int32).view(np.float32)
            entropy_ph   = tf.placeholder(tf.float32, entropy_init.shape)
            entropy_var  = tf.get_variable("entropy", initializer=entropy_ph, trainable=False)
            sess.run(tf.group(entropy_var.initializer), feed_dict={ entropy_ph : entropy_init })
            bs.set_entropy(entropy_var)

            for size in (1024*16*64,): #1024*16*64-1 131072, 1728, 229376, 262144, 28672, 3670016, 442368, 57344, 802816

                # data = list()
                # ebits = 4
                # fbits = 3
                # ebias = 0
                # ebins = 1 << ebits
                # fbins = 1 << fbits
                # for exp in range(ebins-1, -fbits, -1):
                #     for frac in range(fbins-1, -1, -1):
                #         fraction = frac / fbins
                #         f8 = (1 + fraction) * 2**(exp - ebias)
                #         data.append(f8)
                        #print("%2d %.3f %.8e" % (exp-ebias, fraction, f8))
                # cpuX = np.array(data, dtype=np.float32)

                cpuX = np.random.normal(0.0, 1.0, size).astype(np.float32)
                #cpuE = np.random.normal(0.0, 1.0, size).astype(np.float32)

                x = tf.placeholder(tf.float32, cpuX.shape)
                #e = tf.placeholder(tf.float32, cpuE.shape)

                qspec = bs.QuantizeSpec(
                    ebits      = 4,
                    fbits      = 23,
                    stochastic = 0,
                    denorm     = True,
                    frequency  = 1,
                    mode       = 0,
                    bias_pad   = 0,
                    stdv_mul   = 4.0,
                    logfile    = "/home/scott/quant_log.txt",
                )

                y = bs.quantize(x, qspec)

                sess.run(tf.group(*[v.initializer for v in tf.global_variables("quantize")], name="init"))

                devY, = sess.run( [y], { x: cpuX } )
                devY, = sess.run( [y], { x: cpuX } )

                #devY, (devDX,) = sess.run( [y, tf.gradients(y, [x], e)], { x: cpuX, e: cpuE } )

                # print("mean:", np.abs(cpuX).mean())
                # print(" std:", np.abs(cpuX).std())
                # print(" max:", np.abs(cpuX).max())
                # print(" min:", np.abs(cpuX).min())
                for cpu, gpu in (
                    (cpuX, devY),):
                    #(cpuE, devDX),):

                    dif     = np.abs(cpu - gpu)
                    avgval  = np.abs(cpu).mean()
                    maxdif  = dif.max()
                    max_err = maxdif if avgval == 0 else maxdif / avgval
                    l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(cpu).sum())
                    print("size: %7d max_err%%:%12.8f L2_err: %12.10f" % (cpuX.size, 100*max_err, l2_err))

                for i in range(min(20, cpuX.size)):
                    cpu = "0x%08x" % unpack("I", pack("f", cpuX[i]))
                    dev = "0x%08x" % unpack("I", pack("f", devY[i]))
                    print(cpu, dev)


if __name__ == "__main__":
    tf.test.main()

# 0x3f 7 df6d5 0x3f 8 00000
# 0xbf 0 e2fb9 0xbf 1 00000
# 0x3d c 70ace 0x3d d 00000
# 0xbf 8 34e5d 0xbf 8 00000
# 0xbf 2 acf01 0xbf 2 00000
# 0xbf 3 4c906 0xbf 3 00000
# 0x3f 0 3aa08 0x3f 0 00000
# 0xbf c bb567 0xbf c 00000
# 0xbe 9 91431 0xbe a 00000
# 0xbf 9 8dcd1 0xbf a 00000
# 0x3f 4 e0c3b 0x3f 4 00000
# 0x3f 6 65f34 0x3f 6 00000
# 0xbe e 525e5 0xbe e 00000
# 0x3f c 5d974 0x3f d 00000
# 0x3d e bf111 0x3d e 00000
# 0x3e 3 1a3b2 0x3e 4 00000
# 0x3f f f5fdf 0x40 0 00000
# 0x40 1 6b70f 0x40 2 00000
# 0xbe e 9a48e 0xbe e 00000
# 0x3c f e9693 0x3d 0 00000
