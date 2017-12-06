# Blocksparse

The `blocksparse` package contains TensorFlow Ops and corresponding GPU kernels for block-sparse matrix multiplication.  Also included are related ops like edge bias, sparse weight norm and layer norm.

To learn more, see [the launch post on the OpenAI blog](https://blog.openai.com/block-sparse-gpu-kernels/).

## Prerequisites

First, you need at least one Nvidia GPU. For best performance, we recommend using a Pascal or Maxwell generation GPU -- this is the full list of features by GPU type:

| GPU Family | BSMatMul-ASM | BSMatMul-CudaC | BSConv |
|------------|------------------------|----------------|--------|
| Kepler | - | X | - |
| Maxwell | X (fastest) | X | X |
| Pascal | X (fastest) | X | X |
| Volta | - | X (fastest) | - |

Note that BSMatMul-CudaC **only supports `feature_axis=0`**, while BSMatMul-ASM only supports `feature_axis=1`.

Additionally, you need:

- A working Linux installation (we run Ubuntu 16.04) with the Nvidia drivers for your GPU.
- CUDA 8 (in `/usr/local/cuda`)
- Python 3.5 or newer, or 2.7 or newer
- TensorFlow 1.4.0 or newer, [with GPU support](https://www.tensorflow.org/install/install_linux#install_tensorflow) (e.g. `pip install tensorflow-gpu`)
- CUDA 9 and Volta will work if you update the build targets (-gencode=arch=compute_70,code=sm_70) and also build tenorflow from source.

## Installation

```
pip install blocksparse
```

## Usage

This example performs a block-sparse matrix multiplication:
```
from blocksparse.matmul import BlocksparseMatMul
import tensorflow as tf
import numpy as np

hidden_size = 4096
block_size = 32
minibatch_size = 64

# Create a (random) sparsity pattern
sparsity = np.random.randint(2, size=(hidden_size//block_size,hidden_size//block_size))

# Initialize the sparse matrix multiplication object
bsmm = BlocksparseMatMul(sparsity, block_size=block_size)

# Input to graph
x = tf.placeholder(tf.float32, shape=[None, hidden_size])

# Initialize block-sparse weights
w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

# Block-sparse matrix multiplication
y = bsmm(x, w)

# Run
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
result = sess.run([y], feed_dict = {x: np.ones((minibatch_size,hidden_size), dtype='float32')})
print(result)
```

For a more involved example using block-sparse ops to train a language model, see [`examples/`](./examples/).

## Development

If you're interested in hacking on the ops and kernels, go ahead and build from source:

    git clone git@github.com:openai/blocksparse.git
    cd blocksparse

    make compile
    pip install dist/*.whl

    # test it if you like
    test/blocksparse_matmul_test.py
    test/blocksparse_conv_test.py

If your CUDA is not in `/usr/local/cuda` or you have several versions, e.g. both `/usr/local/cuda-8.0` and `/usr/local/cuda-9.0`, set `CUDA_HOME` to the base path to use when compiling `make compile`.


## API Documentation:


### blocksparse.matmul

    class BlocksparseMatMul(object)

        def __init__(self, layout, block_size=32, feature_axis=1, name=None)

        def i_shape(self, N): return (N, self.C) if self.axis else (self.C, N)
        def o_shape(self, N): return (N, self.K) if self.axis else (self.K, N)

        # return the coordinate in the layout that corresponds to a given block id
        def block_coord(self, block): return self.updat_list[block]


        def ortho_init(self)

        def identity_init(self, gpu=False)

        def l2_normalize(self, W, gain=None, epsilon=1e-12, dtype=np.float32)


        def __call__(self, I, W, dw_dtype=tf.float32, name=None, bench=0)

        def group_param_grads(param_grad, group_size=8, cast32=True)


    class SparseProj(object):

        def __init__(self, nhidden, nproj=None, proj_stride=None, block_size=32, gather_lut=None, name=None)

        def gather(self, x)
        def scatter(self, x)
        def scatter_add(self, x, y)
        def scatter_mul(self, x, y)



### blocksparse.conv

    class BlocksparseConv(object):
        """
        BCK: (                                             # block(B)/input(C)/output(K) feature dims
                 ( (c0, c1, c2, ...), (k0, k1, k2, ...) ), # block 0
                 ( (c0, c1, c2, ...), (k0, k1, k2, ...) ), # block 1
                 ( (c0, c1, c2, ...), (k0, k1, k2, ...) ), # block 2 ...
             )
        TRS: (T,R,S) or (R,S) or (S,)         - filter spatial size dims
        DHW: (D,H,W) or (H,W) or (W,)         - input image spatial size dims
        MPQ: (M,P,Q) or (P,Q) or (Q,) or None - output image spatial size dims (used for ambiguous dims in strided transpose conv)
        strides: (1,1,1) or (1,1) or (1,)
        dilates: (1,1,1) or (1,1) or (1,)
        padding: (1,1,1) or (1,1) or (1,) or "SAME" or "VALID"
        edge_bias: True/False
        """
        def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", edge_bias=False, debug=False, deconv=False)

        def edge_bias_shape(self)

        def i_shape(self, N)
        def o_shape(self, N)
        def f_shape(self, block=None)


        def __call__(self, F, I, edge_bias=None):

        def l2_normalize(self, F, gain=None, epsilon=1e-12, dtype=np.float32):

    class BlocksparseDeconv(BlocksparseConv)

        def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", edge_bias=False, debug=False)


    def cwise_linear(x, a=None, b=None)



### blocksparse.ew

    def        add(x, y, name=None)
    def   multiply(x, y, name=None)
    def   subtract(x, y, name=None)
    def     divide(x, y, name=None)
    def    maximum(x, y, name=None)
    def    minimum(x, y, name=None)

    def   negative(x,    name=None)
    def reciprocal(x,    name=None)
    def     square(x,    name=None)
    def       sqrt(x,    name=None)
    def        exp(x,    name=None)
    def        log(x,    name=None)
    def    sigmoid(x,    name=None)
    def       tanh(x,    name=None)
    def       relu(x,    name=None)

    def elu (x, alpha=1.0, name=None)

    def fused_lstm_gates(c, *args, name=None)

    def split4(x)
    def concat4(x0, x1, x2, x3)

    def float_cast(x, dtype, dx_dtype=None)

    def dropout(x, keep_prob=0.8, mask=None)

    def add_n8(xs, name=None)



### blocksparse.norms

    def layer_norm(x, g, b, axis=1, epsilon=1e-6, relu=False, bench=0)

    def batch_norm(x, g, b, epsilon=1e-6)

    def batch_norm_inference(x, g, b, m, v, epsilon=1e-6)


