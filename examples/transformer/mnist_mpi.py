#!/usr/bin/env python

import argparse
import numpy as np
import tensorflow as tf
from tqdm   import tqdm
from mpi4py import MPI
from tensorflow.examples.tutorials.mnist import input_data

from blocksparse.transformer import transpose_0213, masked_softmax
from blocksparse.norms       import layer_norm
from blocksparse.optimize    import Adam
from blocksparse.embed       import embedding_lookup
from blocksparse.quantize    import QuantizeSpec, quantize, set_entropy
from blocksparse.ewops       import bias_relu
from blocksparse.nccl        import allreduce, group_allreduce, sync_variables_op

qspec_e4f3 = QuantizeSpec(
    ebits      = 4,
    fbits      = 3,
    denorm     = True,
    frequency  = 512,
    bias_pad   = 1,
)
qspec_e5f2 = QuantizeSpec(
    ebits      = 5,
    fbits      = 2,
    stochastic = 2,
    denorm     = True,
    frequency  = 512,
    bias_pad   = 8,
)
qspec_e6f7 = QuantizeSpec(
    ebits      = 6,
    fbits      = 7,
    stochastic = 0,
    denorm     = True,
    frequency  = 512,
    bias_pad   = 8,
)

def quantize_pre(x, name, tag):
    if tag != "none":
        if mpi_rank == 0:
            qspec_f = QuantizeSpec(copy=qspec_e4f3, logfile="qspec_e4f03.f.%s.txt" % tag)
            qspec_b = QuantizeSpec(copy=qspec_e6f7, logfile="qspec_e6f07.b.%s.txt" % tag)
        else:
            qspec_f = qspec_e4f3
            qspec_b = qspec_e6f7
        return quantize(x, qspec_f, qspec_b, name=name)
    return x

def quantize_post(x, name, tag):
    if tag != "none":
        if mpi_rank == 0:
            qspec_f = QuantizeSpec(copy=qspec_e6f7, logfile="qspec_e6f07.f.%s.txt" % tag)
            qspec_b = QuantizeSpec(copy=qspec_e5f2, logfile="qspec_e5f02.b.%s.txt" % tag)
        else:
            qspec_f = qspec_e6f7
            qspec_b = qspec_e5f2
        return quantize(x, qspec_f, qspec_b, name=name)
    return x

def layernorm(x, scope, epsilon=1e-5, relu=False):
    """
    normalize state vector to be zero mean / unit variance + learned scale/shift
    """
    n_state = shape_list(x)[-1]
    with tf.variable_scope(scope):
        gain = tf.get_variable('gain', [n_state], initializer=tf.constant_initializer(1))
        bias = tf.get_variable('bias', [n_state], initializer=tf.constant_initializer(0))
        return layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)

def conv1d(x, scope, nf, hps, w_init=tf.random_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(0), relu=False):
    with tf.variable_scope(scope):
        nx    = x.shape[-1].value
        ndims = x.shape.ndims

        w = tf.get_variable("w", [nx, nf], initializer=w_init)
        b = tf.get_variable("b", [    nf], initializer=b_init)

        if ndims > 2:
            y_shape = tf.concat([tf.shape(x)[ : ndims-1], [nf]], axis=0)
            x = tf.reshape(x, [-1, nx])

        scope = tf.get_variable_scope().name
        w = quantize_pre(w, name=scope+"/pre_w", tag=hps.tag)
        x = quantize_pre(x, name=scope+"/pre_x", tag=hps.tag)
        y = tf.matmul(x, w)
        y = quantize_post(y, name=scope+"/post_x", tag=hps.tag)
        y = bias_relu(y, b, relu=relu)

        if ndims > 2:
            y = tf.reshape(y, y_shape)

        return y

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def discretize(x, n_bin):
    """
    discretize inputs for embedding - maps 0 to 1 to n integer bins
    """
    return np.digitize(x, np.linspace(0, 1, n_bin), right=True).astype(np.int32)

def subsample(x):
    """
    attention is n^2 - subsample 28x28 mnist images to 14x14 to speed things up
    """
    return x.reshape(-1, 28, 28)[:, ::2, ::2].reshape(-1, 14*14)

def preprocess(x, n_bin, sub_sample=True):
    """
    subsample and discretize image
    """
    if sub_sample:
        x = subsample(x)
    x = discretize(x, n_bin)
    return x

def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)

def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)

def split_heads(x, n, scope):
    """
    (batch, pixel, state) -> (batch, head, pixel, head_state)
    """
    with tf.name_scope(scope):
        return transpose_0213(split_states(x, n))

def merge_heads(x, scope):
    """
    (batch, head, pixel, head_state) -> (batch, pixel, state)
    """
    with tf.name_scope(scope):
        return merge_states(transpose_0213(x))

def attention(x, scope, n_head, hps):
    """
    perform multi-head qkv dot-product attention and linear project result
    """
    n_state = shape_list(x)[-1]
    with tf.variable_scope(scope):
        q = conv1d(x, 'q', n_state, hps) #project inputs to q,k,v
        k = conv1d(x, 'k', n_state, hps)
        v = conv1d(x, 'v', n_state, hps)
        # c = conv1d(x, 'qkv', n_state*3, hps)
        # q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head, "split_q") #reshape for multi-head attention
        k = split_heads(k, n_head, "split_k")
        v = split_heads(v, n_head, "split_v")

        scope = tf.get_variable_scope().name
        q = quantize_pre(q, name=scope+"/pre_q", tag=hps.tag)
        k = quantize_pre(k, name=scope+"/pre_k", tag=hps.tag)

        with tf.name_scope("qk"):
            w = tf.matmul(q, k, transpose_b=True) #dot product query with key

        w = quantize_post(w, name=scope+"/post_w", tag=hps.tag)

        w = masked_softmax(w, scale=tf.rsqrt(n_state/n_head)) #normalized attention distribution, rescale by head dim

        w = quantize_pre(w, name=scope+"/pre_w", tag=hps.tag)
        v = quantize_pre(v, name=scope+"/pre_v", tag=hps.tag)
        with tf.name_scope("wv"):
            a = tf.matmul(w, v) #reweighted attention value

        a = quantize_post(a, name=scope+"/post_a", tag=hps.tag)

        a = merge_heads(a, "merge") #combine result
        a = conv1d(a, 'proj', n_state, hps) #project result
        return a

def mlp(x, scope, hps, ratio=4):
    """
    2 layer relu residual mlp with wider first layer
    """
    n_state = shape_list(x)[-1]
    with tf.variable_scope(scope):
        hidden   = conv1d(x, 'hidden', n_state*ratio, hps, relu=True) # relu fc layer
        residual = conv1d(hidden, 'residual', n_state, hps) #project back to state size
        return tf.add(x, residual)

def transformer_block(x, scope, n_head):
    """
    core component of transformer
    performs attention + residual mlp + layer normalization
    """
    with tf.variable_scope(scope):
        a = attention(x, 'attention', n_head, hps)
        a = layernorm(tf.add(a, x, name="Add_x"), 'norm_a')
        m = mlp(a, 'mlp', hps)
        m = layernorm(tf.add(m, a, name="Add_a"), 'norm_m')
        return m

def embed_input(x, hps):
    """
    embed discrete inputs to continous space and add learned position embeddings
    """
    x_embed   = tf.get_variable('x_embed',   [hps.n_bin, hps.n_state], initializer=tf.random_normal_initializer(stddev=0.02))
    pos_embed = tf.get_variable('pos_embed', [hps.n_x,   hps.n_state], initializer=tf.random_normal_initializer(stddev=0.01))
    h = tf.add(embedding_lookup(x_embed, x), pos_embed)
    return h

def output(x, hps):
    """
    average pool transformer features and apply linear classifier
    """
    x = tf.reduce_mean(x, axis=1, keepdims=True) #avg pooling features for classifier
    logits = conv1d(x, 'classifier', hps.n_y, hps)[:, 0, :] #squeeze spatial dimension
    return logits

def model(X, Y, hps):

    # tf Variable of random ints of size (3 * GPU_SMs * 1024)
    # tf doesn't support int32 variables?  Hack with float32 view.
    entropy_init = np.random.randint(-(1<<31), (1<<31), size=80*3*1024, dtype=np.int32).view(np.float32)

    if hps.tag != "none":
        qspec_e4f11 = QuantizeSpec(
            ebits      = 4,
            fbits      = 11,
            stochastic = 2,
            denorm     = True,
            frequency  = 512,
            bias_pad   = 1,
            logfile="qspec_e4f11.%s.b.txt" % hps.tag,
        )
        qspec_e5f10 = QuantizeSpec(
            ebits      = 5,
            fbits      = 10,
            stochastic = 2,
            denorm     = True,
            frequency  = 512,
            bias_pad   = 4,
            logfile="qspec_e5f10.%s.b.txt" % hps.tag,
        )
    else:
        qspec_e4f11 = None
        qspec_e5f10 = None
    xs = tf.split(X, mpi_size, 0)
    ys = tf.split(Y, mpi_size, 0)

    with tf.device("/gpu:0"), tf.variable_scope("model"):

        entropy = tf.get_variable("entropy", initializer=entropy_init, trainable=False)
        set_entropy(entropy)

        h = embed_input(xs[mpi_rank], hps)
        for l in range(hps.n_layer):
            h = transformer_block(h, 'layer_%d' % l, hps.n_head)
        logits = output(h, hps)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ys[mpi_rank])
        loss = tf.reduce_mean(loss)

        params = tf.trainable_variables()
        grads  = tf.gradients(loss * cost_scale, params)

        for p in params:
            print(p.op.name + "_" + "_".join(str(x) for x in p.shape.as_list()))

        test = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), ys[mpi_rank]), tf.float32))

        grad_scale = 1.0 / mpi_size

        # all reduce grads
        if mpi_size > 1:
            group_allreduce(grads, params, search_strings=["classifier"] + ["layer_%d" % l for l in range(hps.n_layer-1, -1, -1)], prereduce=8, num_comms=2)

            loss = allreduce(loss) * grad_scale


        train = Adam(grads, params, grad_scale=grad_scale/cost_scale, param_qspec=qspec_e4f11, mean_qspec=qspec_e5f10, var_qspec=qspec_e5f10)

    return loss, train, test

def accuracy(xs, ys, hps, tf_correct):
    """
    compute accuracy over dataset
    """
    n = len(xs)
    correct = 0
    for i in range(0, n, hps.n_batch): #tqdm(, total=n//hps.n_batch, ncols=80, leave=False):
        correct += sess.run(tf_correct, { X: xs[i:i+hps.n_batch], Y: ys[i:i+hps.n_batch] })
    return correct/n


#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':

    np.random.seed(0)
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--n_batch',   type=int, default=512) # combined batch size across all gpus
    parser.add_argument('--n_iter',    type=int, default=100)
    parser.add_argument('--n_bin',     type=int, default=8)
    parser.add_argument('--n_y',       type=int, default=10)
    parser.add_argument('--n_state',   type=int, default=256)
    parser.add_argument('--n_head',    type=int, default=4)
    parser.add_argument('--n_layer',   type=int, default=3)
    parser.add_argument('--profile',   type=int, default=0)
    parser.add_argument('--subsample', type=int, default=1) #2x2 subsampled MNIST
    parser.add_argument('--tag',       type=str, default="") # experiment labal, set to "none" to disable quantization
    hps = parser.parse_args()

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    hps.n_x = 14*14 if hps.subsample else 28*28

    mnist     = input_data.read_data_sets("/home/scott/datasets/mnist")
    n_train   = len(mnist.train.labels)
    n_test    = len(mnist.test.labels)
    ys_train  = mnist.train.labels[:n_test]
    xs_train  = preprocess(mnist.train.images[:n_test], hps.n_bin, hps.subsample)
    xs_test   = preprocess(mnist.test.images, hps.n_bin, hps.subsample)
    ys_test   = mnist.test.labels
    n_updates = hps.n_iter*(n_train//hps.n_batch)

    X = tf.placeholder(tf.int32, [None, hps.n_x])
    Y = tf.placeholder(tf.int32, [None])

    loss, train, tf_correct = model(X, Y, hps)

    config = tf.ConfigProto()
    config.inter_op_parallelism_threads = 1
    config.gpu_options.visible_device_list = str(mpi_rank)
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        if mpi_size > 1:
            sess.run(sync_variables_op(mpi_rank))

        def run(i):
            if hps.profile and i >= hps.profile:
                exit()

            x, y = mnist.train.next_batch(hps.n_batch)
            cost, _ = sess.run([loss, train], { X: preprocess(x, hps.n_bin, hps.subsample), Y: y })

            if i % (n_train//hps.n_batch) == 0 and i > 0:
                train_accuracy = accuracy(xs_train, ys_train, hps, tf_correct)
                test_accuracy  = accuracy(xs_test,  ys_test,  hps, tf_correct)
                if mpi_rank == 0:
                    print("\nupdates %d train accuracy: %.4f test accuracy: %.4f" % (i, train_accuracy, test_accuracy), flush=True)

        print("", flush=True)
        if mpi_rank == 0 and hps.profile == 0:
            for i in tqdm(range(n_updates), total=n_updates, ncols=80, leave=False):
                run(i)
        else:
            for i in range(n_updates):
                run(i)



# m, n, k

# 128*196, 64*4, 64*4
# 128*196, 64*4, 64*4
# 128*196, 64*4, 64*4
#     196,  196,   64  x 128*4
#     196,   64,  196  x 128*4
# 128*196, 64*4, 64*4
# 128*196,256*4, 64*4
# 128*196, 64*4,256*4

# 128*1,    10, 64*4

# 00 q: 256,256 [128 196 256]
# 01 k: 256,256 [128 196 256]
# 02 v: 256,256 [128 196 256]
# 03 attention:qk 2x [128 4 196  64]
# 04 attention:v     [128 4 196 196]
# 05 proj: 256,256 [128 196 256]
# 06 hidden: 256,1024 [128 196 256]
# 07 residual: 1024,256 [128 196 1024]

# 08 q: 256,256 [128 196 256]
# 09 k: 256,256 [128 196 256]
# 10 v: 256,256 [128 196 256]
# 11 attention:qk 2x [128 4 196 64]
# 12 attention:v [128 4 196 196]
# 13 proj: 256,256 [128 196 256]
# 14 hidden: 256,1024 [128 196 256]
# 15 residual: 1024,256 [128 196 1024]

# 16 q: 256,256 [128 196 256]
# 17 k: 256,256 [128 196 256]
# 18 v: 256,256 [128 196 256]
# 19 attention:qk 2x [128 4 196 64]
# 20 attention:v [128 4 196 196]
# 21 proj: 256,256 [128 196 256]
# 22 hidden: 256,1024 [128 196 256]
# 23 residual: 1024,256 [128 196 1024]

# 24 classifier: 256,10 [128 1 256]

q . k.t  = a

QC   . KC.T = QK  16x64   . 16x64.T = 16x16  16x16x64_NT  72,72,16
QK   . KC   = QC  16x16   . 16x64   = 16x64  16x64x16_NN  16,80,80
QK.T . QC   = KC  16x16.T . 16x64   = 16x64  16x64x16_TN  16,80,80

w . v = q

QK   . VC   = QC  16x16   . 16x64   = 16x64  16x64x16_NN
QC   . VC.T = QK  16x64   . 16x64.T = 16x16  16x16x64_NT
QK.T . QC   = VC  16x16.T . 16x64   = 16x64  16x64x16_TN

sequence length = 196
batch size      = 128
head state      = 64
heads           = 4
mlp mult        = 4

      m,    n,    k
128*196, 64*4, 64*4 # q
128*196, 64*4, 64*4 # k
128*196, 64*4, 64*4 # v
    196,  196,   64  x 128*4 # qk (batched matmul)
    196,   64,  196  x 128*4 # wv (batched matmul)
128*196, 64*4, 64*4 # projection
128*196,256*4, 64*4 # mlp
128*196, 64*4,256*4 # mlp



# NC   . CK   = NK
# NK   . CK.T = NC
# NC.T . NK   = CK


# 1 D
B, C, S

B, C/2, 2, S

B, C/2, S, 2

# 2 D
B, C, S

B, H, W, S

B, H/2, W/2, S, 2, 2

B, C/4, S*4