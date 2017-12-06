#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from blocksparse.norms import layer_norm, layer_norm_test, layer_norm_grad_test
import blocksparse.ewops as ew

np.set_printoptions(threshold=8193, linewidth=600, formatter={'int':lambda x: "%4d" % x,'float':lambda x: "%8.6f" % x})

dtypes = [
    (tf.float32,  tf.float32),
    (tf.float16,  tf.float16),
    # (tf.bfloat16, tf.bfloat16),
]

one = 0
out = 0
bench = 0
class LayerNormTest(tf.test.TestCase):

    def testLayerNorm(self):
        # multi-threading screws up benchmarking
        conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with self.test_session(config=conf) as sess, tf.device("/gpu:0"):
            for shape in [ (1024*32, 64), (64,1024*32) ]: # (64,1024*16)
                # assume bigger axis is feature axis
                axis = 0 if shape[0] > shape[1] else 1

                K = shape[  axis]
                N = shape[1-axis]

                if one:
                    X = np.ones(shape, dtype=np.float32)
                    E = np.ones(shape, dtype=np.float32)
                    G = np.ones(    K, dtype=np.float32)
                    B = np.ones(    K, dtype=np.float32)
                    # for n in range(N):
                    #     X[:,n] = np.arange(K)
                else:
                    X = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                    E = np.random.uniform(-1.0, 1.0, shape).astype(np.float32)
                    G = np.random.uniform(-1.0, 1.0,  (K,)).astype(np.float32)
                    B = np.random.uniform(-1.0, 1.0,  (K,)).astype(np.float32)

                x = tf.constant(X)
                e = tf.constant(E)
                g = tf.constant(G)
                b = tf.constant(B)

                for dtF, dtB in dtypes:

                    # just test relu on floats (it's hard to match low precision relu with high precision behavior)
                    relu = dtF is tf.float32 and dtB is tf.float32

                    print("K:%d N:%d Axis:%d Relu:%d F:%s B:%s" % (K, N, axis, relu, dtF.name, dtB.name))

                    Y          = layer_norm_test(X, G, B, axis=axis, relu=relu)
                    DX, DG, DB = layer_norm_grad_test(E, X, G, B, axis=axis, relu=relu)

                    y = ew.float_cast(x, dtype=dtF)
                    y = layer_norm(y, g, b, axis=axis, relu=relu, bench=bench)
                    y = ew.float_cast(y, dtype=tf.float32, dx_dtype=dtB)

                    d = tf.gradients(y, [x, g, b], e)

                    if bench: sess.run(y) #warmup

                    y, (dx, dg, db) = sess.run( [y, d] )

                    for op, cpuA, devA in (
                        (" y:",  Y,  y),
                        ("dx:", DX, dx),
                        ("dg:", DG, dg),
                        ("db:", DB, db),):

                        difA = abs(cpuA - devA)

                        avgval  = np.average(abs(cpuA))
                        maxdif  = difA.max()
                        max_err = maxdif if avgval == 0 else maxdif / avgval

                        l2_err = np.sqrt(np.square(difA).sum()) / np.sqrt(np.square(cpuA).sum())

                        #print("max_err: %5.3f, max_val: %7.3f, l1_err: %7.5f, l2_err: %7.5f" % (difO.max(), cpuO.max(), l1_err, l2_err))

                        print("%s max_err%%:%10.8f L2_err: %12.10f" % (op, 100*max_err, l2_err))

                        # rtol = 1e-4 if dtF is tf.float32 else 1e-1
                        # self.assertAllClose(devA, cpuA, rtol=rtol, atol=rtol)
                        if out:
                            np.savetxt("out.txt",  difA.reshape((-1,N)), fmt='%5.1f')
                            np.savetxt("outC.txt", cpuA.reshape((-1,N)), fmt='%5.1f')
                            np.savetxt("outD.txt", devA.reshape((-1,N)), fmt='%5.1f')
                            exit()
                    print("")

if __name__ == "__main__":
  tf.test.main()



  # 4.87%  308.13us        10  30.812us  30.368us  31.295us  void layer_norm_mean_CN<uint2, int=16>(float*, uint2 const *, int, int, float)
  # 4.62%  292.79us        10  29.279us  28.448us  29.535us  void layer_norm_var_CN<uint2, int=16>(float*, uint2 const *, float4 const *, int, int, float)
  # 1.82%  114.94us        10  11.494us  10.816us  11.840us  void layer_norm_CN<uint2, int=4>(uint2*, uint2 const *, float4 const *, float4 const *, float const *, float const *, int, int, float, int)

  # 3.99%  252.35us        10  25.235us  24.672us  25.696us  void layer_norm_dg_db_CN<float4, uint2>(float*, float*, float4 const *, uint2 const *, float const *, float const *, float4 const *, float4 const *, float, int, int, int)
  # 6.36%  402.78us        10  40.278us  39.839us  41.120us  void layer_norm_dx_sum_CN<float4, uint2, int=16>(float*, float*, float4 const *, uint2 const *, float const *, float const *, float4 const *, float4 const *, float, int, int, int)
  # 4.29%  271.81us        10  27.180us  26.784us  27.424us  void layer_norm_dx_CN<float4, uint2, int=4>(float4*, float4 const *, uint2 const *, float const *, float const *, float4 const *, float4 const *, float4 const *, float4 const *, float, int, int, float, int)


  # 0.36%  22.559us         1  22.559us  22.559us  22.559us  void layer_norm_NC<uint2, float4, int=256>(uint2*, float*, float*, uint2 const *, float4 const *, float4 const *, float, int, float, int)

  # 0.42%  26.560us         1  26.560us  26.560us  26.560us  void layer_norm_dg_db_NC<float, Eigen::half, int=0>(float*, float*, float const *, Eigen::half const *, float const *, float const *, float const *, float const *, float, int, int, int)
  # 0.76%  47.871us         1  47.871us  47.871us  47.871us  void layer_norm_dx_NC<float4, uint2, float4, int=256>(float4*, float4 const *, uint2 const *, float4 const *, float4 const *, float const *, float const *, float, int, float, int)



  # 1.93%  140.58us        10  14.057us  13.632us  14.783us  void layer_norm_mean_CN<float4, int=16, int=256>(float*, float4 const *, int, int, float)
  # 1.58%  114.91us        10  11.491us  11.167us  12.384us  void layer_norm_var_CN<float4, int=16, int=256>(float*, float4 const *, float4 const *, int, int, float)
  # 1.67%  121.50us        10  12.150us  11.744us  12.608us  void layer_norm_CN<float4, int=4>(float4*, float4 const *, float4 const *, float4 const *, float const *, float const *, int, int, float, int)


  # 3.29%  239.10us        10  23.910us  23.295us  24.640us  void layer_norm_dg_db_CN<float4, float4>(float*, float*, float4 const *, float4 const *, float const *, float const *, float4 const *, float4 const *, float, int, int, int)
  # 3.20%  232.80us        10  23.279us  22.880us  23.648us  void layer_norm_dx_CN<float4, float4, int=4>(float4*, float4 const *, float4 const *, float const *, float const *, float4 const *, float4 const *, float4 const *, float4 const *, float, int, int, float, int)
  # 3.05%  222.08us        10  22.207us  21.888us  22.784us  void layer_norm_dx_sum_CN<float4, float4, int=16, int=256>(float*, float*, float4 const *, float4 const *, float const *, float const *, float4 const *, float4 const *, float, int, int, int)
  # 0.86%  62.400us        41  1.5210us     672ns  4.6080us  [CUDA memset]

  # 0.51%  37.088us         1  37.088us  37.088us  37.088us  void layer_norm_NC<float4, float4, int=256>(float4*, float*, float*, float4 const *, float4 const *, float4 const *, float, int, float, int)

  # 0.61%  44.639us         1  44.639us  44.639us  44.639us  void layer_norm_dx_NC<float4, float4, float4, int=256>(float4*, float4 const *, float4 const *, float4 const *, float4 const *, float const *, float const *, float, int, float, int)
  # 0.39%  28.576us         1  28.576us  28.576us  28.576us  void layer_norm_dg_db_NC<float, float, int=0>(float*, float*, float const *, float const *, float const *, float const *, float const *, float const *, float, int, int, int)

