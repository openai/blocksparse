#!/usr/bin/env python

'''
Example of the blocksparse transformer on enwik8.

To download data:

wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip -d /tmp
'''

import argparse
import numpy       as np
import tensorflow  as tf
import blocksparse as bs
from mpi4py import MPI

def layernorm(x, scope, epsilon=1e-5, relu=False):
    """
    normalize state vector to be zero mean / unit variance + learned scale/shift
    """
    n_state = x.shape[-1].value
    with tf.variable_scope(scope):
        gain = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0.0))
        return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)


def conv1d(x, scope, nf, std=0.02, relu=False, fast_gelu=False):
    with tf.variable_scope(scope):
        nx    = x.shape[-1].value
        ndims = x.shape.ndims

        # Note: param initializers are not particularly well tuned in this code
        w = tf.get_variable("w", [nx, nf], initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable("b", [    nf], initializer=tf.constant_initializer(0.0))

        if hps.float16:
            # We delay weight casting till just before use to minimize memory footprint.
            # In recompute mode these casts are released just after use on forward pass,
            # then remade on the recompute pass.
            with tf.control_dependencies([x.op]):
                # By setting dx_dtype to float16 we prevent useless casting back to fp32 in the backwards pass.
                # Our all-reduce and fused optimizers can accept fp16 natively.
                w = bs.float_cast(w, dtype=tf.float16, dx_dtype=tf.float16)

        # merge context and batch dims for more efficient matmul
        if ndims > 2:
            y_shape = tf.concat([tf.shape(x)[: ndims - 1], [nf]], axis=0)
            x = tf.reshape(x, [-1, nx])

        y = tf.matmul(x, w)

        # avoid atomics in bias grad, but be careful as tf handles temp memory badly in the presense of async ops like all-reduce
        y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

        if ndims > 2:
            y = tf.reshape(y, y_shape)

        return y

# Fine sparse structure
# Within each block this mask is applied to force the softmax output to zero where the mask is zero
# This is defined as a callback to avoid having to instantiate the full mask in memory at one time.
# The callback value is immediately converted to a bit mask internally.
def causal_subblock_mask(blk_shape, head_idx, query_idx, key_idx, blk_idx):
    """Prohibit positions in sub-blocks from attending to indices in the future.
    Note: query_idx and key_idx are absolute indices rather than relative to
    each block.
    """
    mask = np.ones(blk_shape, dtype=np.bool)
    if query_idx == key_idx:
        for q, k in np.ndindex(blk_shape):
            if k > q:
                mask[q, k] = 0
    return mask

# Coarse sparse structure
# Only layout[q,k] == 1 blocks are computed and materialized in memory
# Block sizes of 8, 16, 32 and 64 are supported on volta fp16 tensorcores (64 being most appropriate for dense attention)
# Only blocksize 32 currently supported in fp32 on other gpus (sm >= 3.5).
def get_blocksparse_transformer(n_timesteps, n_heads):
    blocksize = 64 if hps.float16 else 32
    n_time_blocks = n_timesteps // blocksize
    # The block layout can also include a head dimension if you don't want the same layout shared by all heads.
    # Each head just has to have the same number of active blocks (but you can always mask them away).
    layout = np.ones([n_time_blocks, n_time_blocks], dtype=np.bool)
    # No query blocks may attend to key blocks in the future.
    # Much more elaborate structures can be defined here aside from the usual lower triangular.
    for q_idx, k_idx in np.ndindex(n_time_blocks, n_time_blocks):
        if k_idx > q_idx:
            layout[q_idx, k_idx] = 0
    bst = bs.BlocksparseTransformer(layout, block_size=blocksize, mask_callback=causal_subblock_mask, heads=n_heads)
    return bst

# very simple to use recompute decorator.  Be sure to pair with bs.gradients() for it to work
@bs.recomputable
def transformer_block(x, scope, train=False):
    """
    core component of transformer
    performs attention + residual mlp + layer normalization
    """
    n_state = x.shape[-1].value

    with tf.variable_scope(scope):

        h = layernorm(x, "norm_a")

        q = conv1d(h, 'proj_q', n_state)
        k = conv1d(h, 'proj_k', n_state)
        v = conv1d(h, 'proj_v', n_state)

        # only need to create one bst per config
        # we could pass this in as an external param but I like to keep the code more local
        bst_params = (hps.n_timesteps, hps.n_head)
        bst = bst_cache.get(bst_params)
        if bst is None:
            bst = bst_cache[bst_params] = get_blocksparse_transformer(*bst_params)

        # run the core bst ops, transposes for dealing with heads are fused in here.
        w = bst.query_key_op(q, k)
        w = bst.masked_softmax(w, scale=1.0/np.sqrt(n_state / hps.n_head))
        a = bst.weight_value_op(w, v)

        a = conv1d(a, 'proj_a', n_state, std=0.02/hps.n_layer)

        if train and hps.resid_pdrop > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_a"
            a, dropout_cache[key] = bs.dropout(a, keep_prob=1.0 - hps.resid_pdrop, mask=dropout_cache.get(key))

        # many basic tf ops are about half as fast as they should be in fp16
        x = bs.add(x, a)

        m = layernorm(x, "norm_m")

        # fast_gelu: x * sigmoid(1.702 * x)
        m = conv1d(m, 'proj_m1', n_state * hps.mlp_ratio, fast_gelu=True)
        m = conv1d(m, 'proj_m2', n_state)

        if train and hps.resid_pdrop > 0.0:
            # preserve the dropout mask through recompute
            key = scope + "_dropout_m"
            m, dropout_cache[key] = bs.dropout(m, keep_prob=1.0 - hps.resid_pdrop, mask=dropout_cache.get(key))

        return bs.add(x, m)


def model(xs, ys, loss_scale=None, train=False):

    with tf.variable_scope("model", reuse=not train):

        with tf.device("/cpu:0"):
            if train:
                grad_scale    = tf.reciprocal(loss_scale) if hps.float16 else 1.0
                global_step   = tf.get_variable("global_step", [], initializer=tf.ones_initializer(), trainable=False)
                learning_rate = tf.minimum(global_step * (1.0/hps.warmup_iters), 1.0) * hps.lr
            mpi_scale = tf.constant(1.0 / mpi_size)

        with tf.device("/gpu:0"):

            # Contains scope/var_name substrings we use to group gradients for all reduce
            # You'll want to find groupings that are scheduled uniquely by tensorflow, otherwise bs.allreduce could hang.
            # The groups should be ordered in which the all-reduce is called.
            # Any gradients not matching the substrings will get appended to the last group.
            grad_groups = []

            # embed discrete inputs to continous space and add learned position embeddings
            with tf.variable_scope('embed'):
                x_embed = tf.get_variable("x",   [   hps.n_vocab,     hps.n_state], initializer=tf.random_normal_initializer(stddev=0.02))
                p_embed = tf.get_variable('pos', [1, hps.n_timesteps, hps.n_state], initializer=tf.random_normal_initializer(stddev=0.01))

                if hps.float16:
                    x_embed = bs.float_cast(x_embed, dtype=tf.float16, dx_dtype=tf.float16)
                    p_embed = bs.float_cast(p_embed, dtype=tf.float16, dx_dtype=tf.float16)

                # bs.embedding_lookup can be much faster than tf version for low entropy indexes or small vocabs
                x = bs.embedding_lookup(x_embed, xs)

                if train and hps.embed_pdrop > 0.0:
                    # this part of the code is not recomputed so no need to remember the generated mask returned by bs.dropout
                    x,       _ = bs.dropout(x,       keep_prob=1.0 - hps.embed_pdrop)
                    p_embed, _ = bs.dropout(p_embed, keep_prob=1.0 - hps.embed_pdrop)

                h = x + p_embed
                grad_groups.insert(0, 'embed')

            for l in range(hps.n_layer):
                layer_name = 'layer_%d' % l
                # enable the recompute decorator in training
                # see blocksparse/grads.py if you want understand how this works
                h = transformer_block(h, layer_name, train=train, recompute=train and hps.recompute)
                grad_groups.insert(0, layer_name)

            #average pool transformer features and apply linear classifier
            with tf.variable_scope('logits'):
                h = tf.reshape(h, [-1, hps.n_state])
                logits = tf.matmul(h, x_embed, transpose_b=True)

            if hps.float16:
                # much faster and more memory efficient (but currently only implemented in fp16)
                loss   = bs.softmax_cross_entropy(logits=logits, labels=ys)
            else:
                labels = tf.cast(tf.reshape(ys, [-1]), tf.int32)
                loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            loss = tf.reduce_mean(loss)

            if train:

                # apply loss scaling in fp16 mode
                if hps.float16:
                    grad_loss = bs.scale_tensor(loss, loss_scale)
                else:
                    grad_loss = loss

                # use bs.gradients to allow bs.recomputable decorators to work
                params = tf.trainable_variables()
                grads  = bs.gradients(grad_loss, params)

                if mpi_size > 1:
                    # apply (1.0 / mpi_size) scaling prior to all_reduce to allow greater utilization of fp16 dynamic range.
                    # That is we're ok with flushing some small values to zero to allow growth of large values in allreduce (without hitting inf).
                    loss  = bs.scale_tensor(loss, mpi_scale)
                    grads = [bs.scale_tensor(g, mpi_scale) for g in grads]

                    # allreduce in an mpi context
                    # bias and gain grads will be in fp32, but have them fp16 cast prior to allreduce
                    cast_all = tf.float16 if H.float16 else None
                    loss  = bs.allreduce(loss)
                    grads = bs.group_allreduce(grads, params, search_strings=grad_groups, cast_all=cast_all)

                # This does not actually perform the clippiing, only measures the norm_scale needed to be applied.
                # norm_scale is then later applied in the fused optimizer ops (eliminating an extra pass over the gradients).
                # norm_scale is also used to detect inf/nan values in any of the gradients so the whole update can be skipped
                # and tried again with a new loss_scale.
                global_norm, norm_scale = bs.clip_by_global_norm(grads, grad_scale=grad_scale, clip_norm=hps.clip_norm)

                # Apply AdamOptimizer:
                # fp16 mode is a special feature to store running mean and variance variables in custom fp16 formats.
                # Using this mode should incure no loss in accuracy and save a lot of memory in your model.
                # For futher memory savings consider using bs.AdafactorOptimizer.
                adam = bs.AdamOptimizer(learning_rate=learning_rate, norm_scale=norm_scale, grad_scale=grad_scale, fp16=hps.float16)

                train_op = adam.apply_gradients(zip(grads, params))

                # update global step after we're done using it for this update
                with tf.control_dependencies([ train_op ]), tf.device("/cpu:0"):
                    update_op = tf.assign_add(global_step, 1.0)

                return loss, tf.group(train_op, update_op), global_norm, norm_scale

            else:
                if mpi_size > 1:
                    loss = bs.allreduce(bs.scale_tensor(loss, mpi_scale))

                return loss



def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    X = np.fromstring(open(path).read(n_train + n_valid + n_test), dtype=np.uint8)
    trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return trX, vaX, teX


def iter_data(X, n_timesteps, n_batch, mpi_rank, mpi_size):
    offset = np.random.randint(0, n_timesteps)
    idxs   = np.random.permutation(np.arange(offset, X.size - (n_timesteps + 1), n_timesteps))
    # Truncate the training set this epoch if it does not divide evenly
    sequences_per_batch = mpi_size * n_batch
    length = (idxs.size // sequences_per_batch) * sequences_per_batch
    if length != idxs.size:
        print_rank0('Not including {} sequences'.format(idxs.size - length))
    idxs = idxs[:length]
    # Reshape starting indices to K*mpi_size*n_batch
    idxs = idxs.reshape([-1, mpi_size, n_batch])
    print_rank0(f'Number of minibatches this epoch: {len(idxs)}')
    for minibatch_index in range(len(idxs)):
        starting_indices = idxs[minibatch_index, mpi_rank]
        x = np.zeros((n_batch, n_timesteps + 1), dtype=np.uint8)
        for i, start_idx in enumerate(starting_indices):
            x[i, :] = X[start_idx:start_idx + n_timesteps + 1]
        yield x[:, :-1], x[:, 1:]


def print_rank0(*args):
    if mpi_rank == 0:
        print(*args, flush=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs',     type=int,   default=100)
    parser.add_argument('--n_batch',      type=int,   default=32)
    parser.add_argument('--n_state',      type=int,   default=512)
    parser.add_argument('--n_head',       type=int,   default=4)
    parser.add_argument('--n_layer',      type=int,   default=6)
    parser.add_argument('--n_timesteps',  type=int,   default=320)
    parser.add_argument('--n_vocab',      type=int,   default=256)
    parser.add_argument('--mlp_ratio',    type=int,   default=4)
    parser.add_argument('--lr',           type=float, default=0.0005)
    parser.add_argument('--resid_pdrop',  type=float, default=0.05)
    parser.add_argument('--embed_pdrop',  type=float, default=0.05)
    parser.add_argument('--clip_norm',    type=float, default=1.0)
    parser.add_argument('--loss_scale',   type=float, default=2.0**16)
    parser.add_argument('--loss_count',   type=int,   default=1000)
    parser.add_argument('--warmup_iters', type=int,   default=1000)
    parser.add_argument('--enwik8_path',  type=str,   default='/home/scott/datasets/enwik8') # obviously change to your local path
    parser.add_argument('--log_interval', type=int,   default=200)
    parser.add_argument('--profile',      type=int,   default=0) # exit early for nvprof profiling
    parser.add_argument('--float16',      type=int,   default=0) # only sm >= 7.0 (tensorcores)
    parser.add_argument('--recompute',    type=int,   default=0) # allow use of large contexts and/or lots of layers/params

    # use some global vars for convenience
    hps = parser.parse_args()

    bst_cache     = dict()
    dropout_cache = dict()

    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()

    n_train = int(90e6)
    n_valid = int(5e6)
    n_test  = int(5e6)
    trainX, validX, testX = enwik8(hps.enwik8_path, n_train, n_valid, n_test)

    with tf.device("/gpu:0"):
        X = tf.placeholder(tf.uint8, shape=[hps.n_batch, hps.n_timesteps])
        Y = tf.placeholder(tf.uint8, shape=[hps.n_batch, hps.n_timesteps])

    # loss_scale is a host side scalar
    with tf.device("/cpu:0"):
        loss_scale = tf.placeholder(tf.float32, shape=[])

    # needed for bs.dropout()
    np.random.seed(mpi_rank)
    bs.set_entropy()

    # initialize the loss_scale placeholder value
    cur_loss_scale = hps.loss_scale
    loss_count = 0

    # build the models for training and testing/validation
    train_loss, train_op, gn, ns = model(X, Y, loss_scale, train=True)
    valid_loss = model(X, Y)

    # Free up some python memory now that models are built
    bst_cache     = None
    dropout_cache = None
    bs.clear_bst_constants()

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(mpi_rank)
    config.allow_soft_placement = True

    iteration = 0
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        if mpi_size > 1:
            # sync variables initialized on rank 0 to all other ranks
            sess.run(bs.sync_variables_op(mpi_rank))

        for i in range(hps.n_epochs):
            print_rank0(f'Starting epoch {i}')
            for x, y in iter_data(trainX, hps.n_timesteps, hps.n_batch, mpi_rank, mpi_size):

                retry = True
                while retry:

                    loss, global_norm, norm_scale, _ = sess.run([train_loss, gn, ns, train_op], feed_dict={X: x, Y: y, loss_scale: cur_loss_scale})

                    # auto loss scaling for fp16.
                    if hps.float16 and np.isfinite(loss):
                        # slowly increase loss scale but quickly drop it when inf or nan is detected in the gradients
                        # norm_scale will be zero when this happens
                        # You may also want to limit the change in loss_scale from any single minibatch and throw them away when this limit is exceeded.
                        if norm_scale == 0.0:
                            cur_loss_scale *= 0.5
                            loss_count      = 0
                            print_rank0("fp16 saturation detected (%f), changing loss_scale to: 2^%.0f" % (global_norm, np.log2(cur_loss_scale)))
                        else:
                            retry = False
                            if loss_count >= hps.loss_count:
                                cur_loss_scale *= 2.0
                                loss_count      = 0
                                print_rank0("No fp16 saturation detected after %d iterations, changing loss_scale to: 2^%.0f" % (hps.loss_count, np.log2(cur_loss_scale)))
                            else:
                                loss_count += 1
                    else:
                        # if forward pass is not finite skip any further auto loss scaling.
                        retry = False

                if iteration % hps.log_interval == 0:
                    print_rank0('train iteration: %7d, loss: %.5f, bits per byte: %.5f ns:%.5f gn:%.5f' % (iteration, loss, loss/np.log(2), norm_scale, global_norm))
                iteration += 1

                if hps.profile and iteration >= hps.profile:
                    exit()


            print_rank0('Calculating validation loss')
            valid_losses = []
            for x, y in iter_data(validX, hps.n_timesteps, hps.n_batch, mpi_rank, mpi_size):

                valid_losses.append(sess.run(valid_loss, feed_dict={X: x, Y: y}))

            avg_valid = sum(valid_losses) / len(valid_losses)
            print_rank0('Average validation loss: %.5f, bits per byte: %.5f' % (avg_valid, avg_valid/np.log(2)))


        print_rank0('Calculating test loss')
        test_losses = []
        for x, y in iter_data(testX, hps.n_timesteps, hps.n_batch, mpi_rank, mpi_size):

            test_losses.append(sess.run(valid_loss, feed_dict={X: x, Y: y}))

        avg_test = sum(test_losses) / len(test_losses)
        print_rank0('Average test loss: %.5f, bits per byte: %.5f' % (avg_test, avg_test/np.log(2)))
