#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs

shapes = [
    # [  4,      4 ],
    # [ 60,     60 ],
    # [ 64,     64 ],
    # [ 64,    256 ],
    # [ 256,    64 ],
    [ 4096, 4096*8 ],
]

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)

class TransposeTest(tf.test.TestCase):

    def testTranspose(self):
        with self.test_session(config=config) as sess, tf.device("/gpu:0"):

                for shape in shapes:

                    cpuX = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                    x = tf.placeholder(tf.float32, shape, name="x")

                    for dtype in (tf.float16, tf.float32):  #tf.float16, tf.float32

                        xf = bs.float_cast(x, dtype=dtype)

                        y = bs.transpose_2d(xf)
                        y = bs.float_cast(y, dtype=tf.float32)

                        Y = tf.transpose(xf)
                        Y = bs.float_cast(Y, dtype=tf.float32)

                        y, Y = sess.run( [ y, Y ], feed_dict={ x : cpuX } )

                        dif     = np.abs(Y - y)
                        avgval  = np.average(abs(Y))
                        maxdif  = dif.max()
                        max_err = maxdif if avgval == 0 else maxdif / avgval
                        l2_err  = np.sqrt(np.square(dif).sum()) / np.sqrt(np.square(Y).sum())

                        print("%s, shape:%16s, err:%17.12f, l2_err:%17.12f" % (dtype.name, str(shape), max_err, l2_err))



if __name__ == "__main__":
  tf.test.main()
