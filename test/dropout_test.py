#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from tensorflow.python.ops import gradient_checker

def ceil_div(x, y):
    return -(-x // y)

shapes = [
    # [ [32, 32], [ [32, 1] ] ],
    [ [  64,], [ None, ] ],
    [ [1024,], [ None, ] ],
    [ [1023,], [ None, ] ],
    [ [1024, 128], [ [1024, 1], [1, 128], None ] ],
    [ [1023, 127], [ [1023, 1], [1, 127], None ] ],
    [ [64, 64, 64], [ [64, 64, 1], [64, 1, 64], [1,64,64], [1,64,1], [64,1,1], [1,1,64], [1,1,1], None ] ],
    [ [63, 63, 63], [ [63, 63, 1], [63, 1, 63], [1,63,63], [1,63,1], [63,1,1], [1,1,63], [1,1,1], None ] ],
    [ [16,16,16,16,16], [ [16,16,16,16,1], None ] ],
]

class DropoutTest(tf.test.TestCase):

    def testDropout(self):

        config = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=config) as sess:

            bs.set_entropy()
            sess.run(tf.global_variables_initializer())

            # with tf.device("/gpu:0"):
            #     x = tf.ones([10000])*-10.0
            #     g = bs.concrete_gate(x)
            #     g = sess.run(g)
            #     print(g.sum()/g.size)

                # error = gradient_checker.compute_gradient_error(x, x.shape, g, g.shape) #, extra_feed_dict={ x: cpuX, m: mask }
                # print(error)

            for dtype in (tf.float16, ):  #tf.float16, tf.bfloat16
                for x_shape, mask_shapes in shapes:
                    for mask_shape in mask_shapes:

                        m_shape = x_shape if mask_shape is None else mask_shape

                        cpuO = np.ones(x_shape, dtype=np.float32)
                        cpuX = np.random.uniform(-1.0, 1.0, x_shape).astype(np.float16).astype(np.float32)
                        cpuM = np.random.randint(0, 2, size=m_shape, dtype=np.bool)

                        mask = np.zeros(ceil_div(cpuM.size, 32)*32, dtype=np.bool)
                        mask[:cpuM.size] = cpuM.reshape(-1)
                        mask = np.packbits(mask.reshape(-1,8)[:,::-1]).view(np.int32)

                        cpuY = cpuX * cpuM.astype(np.float32) * 2.0

                        with tf.device("/gpu:0"):

                            x = tf.placeholder(tf.float32, cpuX.shape)
                            m = tf.placeholder(tf.int32,   mask.shape)

                            xf   = bs.float_cast(x, dtype=dtype)
                            y, _ = bs.dropout(xf, keep_prob=0.5, mask=m, mask_shape=mask_shape)
                            y    = bs.float_cast(y, dtype=tf.float32)

                            devY, = sess.run( [y,], feed_dict={ x: cpuX, m: mask } )

                            xf   = bs.float_cast(x, dtype=dtype)
                            y, _ = bs.dropout(xf, keep_prob=0.8, mask_shape=mask_shape)
                            y    = bs.float_cast(y, dtype=tf.float32)

                            devO, = sess.run( [y,], feed_dict={ x: cpuO } )


                        diff = np.abs(devY - cpuY)
                        print("dype: %8s x_shape: %-20s m_shape: %-20s err: %4.2f norm_sum: %4.2f" % ( dtype.name, str(x_shape), str(mask_shape), diff.sum(), devO.sum()/devO.size ))
                        #np.savetxt( "diff.txt",  diff, fmt="%4.2f")


if __name__ == "__main__":
  tf.test.main()
