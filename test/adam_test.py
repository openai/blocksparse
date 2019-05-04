#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from blocksparse.optimize import adam_op

ones        = 0
out         = 0
beta1       = 0.8
beta2       = 0.5
learn_rate  = 0.5
grad_scale  = 1.0
clip_thresh = 1.0
clip_norm   = 1.0
clip_sigma  = 0.0
epsilon     = 1e-8

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)


class AdafactorTest(tf.test.TestCase):

    def testAdafactor(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for dtype in (tf.float32, tf.float16): # tf.float16
                for shape in (
                     (1,),
                     (3,),
                     (127),
                     (1,1024),
                     (1023,1024),
                     (1024,1024),
                ):
                    if ones:
                        G = np.ones( shape, dtype=np.float32)
                        P = np.ones( shape, dtype=np.float32)
                        M = np.zeros(shape, dtype=np.float32)
                        V = np.zeros(shape, dtype=np.float32)
                    else:
                        G = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                        P = np.random.uniform(-1.0, 1.0, shape).astype(np.float16).astype(np.float32)
                        M = np.random.uniform( 0.0, 1.0, shape).astype(np.float16).astype(np.float32)
                        V = np.random.uniform( 0.0, 1.0, shape).astype(np.float16).astype(np.float32)

                    g = tf.placeholder(tf.float32, G.shape)
                    p = tf.Variable(initial_value=P, name="p")
                    m = tf.Variable(initial_value=M, name="m")
                    v = tf.Variable(initial_value=V, name="v")
                    sess.run( tf.global_variables_initializer() )

                    g = bs.float_cast(g, dtype=dtype)

                    global_norm, norm_scale = bs.clip_by_global_norm([g], grad_scale=grad_scale, clip_norm=clip_norm)

                    p, m, v = sess.run(
                        adam_op(
                            g, p, m, v, learn_rate, grad_scale, clip_sigma, [norm_scale], [],
                            decay_mean=beta1, decay_var=beta2, epsilon=epsilon),
                        feed_dict={ g: G } )

                    GN = np.sqrt(np.sum(np.square(G*grad_scale), keepdims=True))
                    NS = clip_norm / np.maximum(GN, clip_norm)
                    G *= NS * grad_scale


                    M = beta1 * M +  (1.0 - beta1) * G
                    V = beta2 * V +  (1.0 - beta2) * G*G

                    P -= learn_rate * M / (np.sqrt(V) + epsilon)

                    print("testAdam", dtype, GN, NS)
                    for op, dev, cpu in [
                        [ "M", m, M ],
                        [ "V", v, V ],
                        [ "P", p, P ],
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
            np.savetxt("out_%s_dif.txt"%op, dif, fmt='%6.3f')
            np.savetxt("out_%s_cpu.txt"%op, cpu, fmt='%6.3f')
            np.savetxt("out_%s_gpu.txt"%op, dev, fmt='%6.3f')


if __name__ == "__main__":
  tf.test.main()

# a = np.zeros((32,32), dtype=np.bool)
# for y, x in np.ndindex(a.shape):
#     if x <= y: a[y,x] = True
# b = np.packbits(a.reshape(-1,8)[:,::-1]).view(np.uint32)

# np.unpackbits(b.view(np.uint8))

# b = np.packbits(a.reshape(-1,8)[:,::-1])