#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from blocksparse.optimize import adafactor1d_op, adafactor2d_op

ones        = 0
out         = 0
beta2       = 0.5
learn_rate  = 0.5
grad_scale  = 1.0
clip_thresh = 1.0
clip_norm   = 1.0
epsilon     = 1e-30

config = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)


class AdafactorTest(tf.test.TestCase):

    def testAdafactor(self):

        with self.test_session(config=config) as sess, tf.device("/gpu:0"):
            for dtype in (tf.float32, tf.float16): # tf.float16
                for shape_g in (
                     (1024,1024*2),
                     (   1,1024*2),
                     (1024,1023*1),
                     (   1,1023*1),
                ):

                    shape_c = ( 1,shape_g[1])
                    shape_r = (shape_g[0], 1)

                    if ones:
                        G = np.ones( shape_g, dtype=np.float32)
                        P = np.ones( shape_g, dtype=np.float32)
                        C = np.zeros(shape_c, dtype=np.float32)
                        R = np.zeros(shape_r, dtype=np.float32)
                    else:
                        G = np.random.uniform(-1.0, 1.0, shape_g).astype(np.float16).astype(np.float32)
                        P = np.random.uniform(-1.0, 1.0, shape_g).astype(np.float16).astype(np.float32)
                        C = np.random.uniform( 0.0, 1.0, shape_c).astype(np.float16).astype(np.float32)
                        R = np.random.uniform( 0.0, 1.0, shape_r).astype(np.float16).astype(np.float32)

                    g = tf.placeholder(tf.float32, G.shape)
                    p = tf.Variable(initial_value=P, name="p")
                    c = tf.Variable(initial_value=C, name="c")
                    r = tf.Variable(initial_value=R, name="r")
                    sess.run( tf.global_variables_initializer() )

                    g = bs.float_cast(g, dtype=dtype)

                    # adafactor has it's own fused infinity filtering but quick test of this standalone op here.
                    g = bs.filter_tensor(g)

                    global_norm, norm_scale = bs.clip_by_global_norm([g], grad_scale=grad_scale, clip_norm=clip_norm)

                    if shape_g[0] > 1:

                        p, c, r, x, _ = sess.run(
                            adafactor2d_op(p, c, r, g, beta2, learn_rate, grad_scale, clip_thresh, [norm_scale], epsilon=epsilon),
                            feed_dict={ g: G } )

                        GN = np.sqrt(np.sum(np.square(G*grad_scale), keepdims=True))
                        NS = clip_norm / np.maximum(GN, clip_norm)
                        G *= NS * grad_scale

                        C = beta2 * C + (1.0 - beta2) * np.mean(np.square(G) + epsilon, axis=0, keepdims=True)
                        R = beta2 * R + (1.0 - beta2) * np.mean(np.square(G) + epsilon, axis=1, keepdims=True)
                        LTM = np.mean(R, keepdims=True)
                        X = G / (np.sqrt(R / LTM) * np.sqrt(C))
                        RMS_X = np.sqrt(np.mean(np.square(X), keepdims=True))

                    else:

                        r = R
                        p, c, x, _ = sess.run(
                            adafactor1d_op(p, c, g, beta2, learn_rate, grad_scale, clip_thresh, [norm_scale], epsilon=epsilon),
                            feed_dict={ g: G } )

                        GN = np.sqrt(np.sum(np.square(G*grad_scale), keepdims=True))
                        NS = clip_norm / np.maximum(GN, clip_norm)
                        G *= NS * grad_scale

                        C = beta2 * C + (1.0 - beta2) * (np.square(G) + epsilon)
                        X = G / np.sqrt(C)
                        RMS_X = np.sqrt(np.mean(np.square(X), keepdims=True))

                    P -= learn_rate * X / np.maximum(1.0, RMS_X / clip_thresh)

                    print("testAdafactor", dtype, GN, NS)
                    for op, dev, cpu in [
                        [ "C", c, C ],
                        [ "R", r, R ],
                        [ "X", x, X ],
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
            np.savetxt("%s_dif.txt"%op, dif, fmt='%6.3f')
            np.savetxt("%s_cpu.txt"%op, cpu, fmt='%6.3f')
            np.savetxt("%s_gpu.txt"%op, dev, fmt='%6.3f')


if __name__ == "__main__":
  tf.test.main()

# a = np.zeros((32,32), dtype=np.bool)
# for y, x in np.ndindex(a.shape):
#     if x <= y: a[y,x] = True
# b = np.packbits(a.reshape(-1,8)[:,::-1]).view(np.uint32)

# np.unpackbits(b.view(np.uint8))

# b = np.packbits(a.reshape(-1,8)[:,::-1])