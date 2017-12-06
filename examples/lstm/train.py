#!/usr/bin/env python

# import memory_util as mu
# mu.vlog(1)

import os
import time
import argparse
import logging
import platform
import numpy      as np
import tensorflow as tf
from tqdm   import tqdm
import layers
from layers import HParams, LSTM_Model
from utils  import text8, text8_stream, wiki3, wiki3_stream, num_trainable_params, print_trainable_params, make_path, JsonLogger
from tensorflow.contrib import nccl
#from tensorflow.python.client import timeline


def model(X, S, Y, hps, train=False, ema=None):

    xs = tf.split(X, hps.ngpu, 1)
    ys = tf.split(Y, hps.ngpu, 1)
    ss = tf.split(S, hps.ngpu, 2 - hps.axis)

    losses = []
    states = []
    grads  = []
    for gpu in range(hps.ngpu):
        with tf.device("/gpu:%d" % gpu), tf.variable_scope("model%d" % gpu, reuse=not train):
            lstm_model  = LSTM_Model(hps, train)
            loss, state = lstm_model.forward(xs[gpu], ss[gpu], ys[gpu], ema=ema)
            losses.append(loss)
            states.append(state)
            if train:
                grads.append( lstm_model.backward() )

    if train:
        ngrads = len(grads[0])
        if hps.ngpu > 1:
            # all reduce grads
            for i in range(ngrads):

                sum_grads = nccl.all_sum( [ grads[gpu][i][0] for gpu in range(hps.ngpu) ] )
                for gpu in range(hps.ngpu):
                    grads[gpu][i] = ( sum_grads[gpu], grads[gpu][i][1] )

        train = list()
        for gpu, gpu_grads in enumerate(grads):
            with tf.device("/gpu:%d" % gpu), tf.variable_scope("opt%d" % gpu):

                # compute average from sum
                if hps.ngpu > 1:
                    for i in range(ngrads):
                        # Note the scalar division must appear in a device context otherwise
                        # it will do a whole lot unnecessary of gpu to gpu copying.
                        # Also rebuild the tuple.
                        gpu_grads[i] = ( gpu_grads[i][0]/float(hps.ngpu), gpu_grads[i][1] )

                if hps.optimizer == 'adam_old':
                    trainer = tf.train.AdamOptimizer(learning_rate=hps.lr, beta2=hps.beta2)
                    train.append(trainer.apply_gradients(gpu_grads))
                else:
                    param_grads = [gpu_grads[i][0] for i in range(ngrads)]
                    param_names = [gpu_grads[i][1] for i in range(ngrads)]
                    if hps.optimizer == 'adam':
                        train.append(layers.adam_updates(param_names, param_grads, lr=hps.lr, mom2=hps.beta2, gamma=hps.gamma))
                    if hps.optimizer == 'adamax':
                        train.append(layers.adamax_updates(param_names, param_grads, lr=hps.lr, mom2=hps.beta2))

        train = tf.group(*train)
    else:
        train = None

    states = tf.concat(states, 2 - hps.axis)

    return train, tf.add_n(losses)/hps.ngpu, states

def score(text, hps):
    smb = np.zeros(hps.state_shape)
    costs = []
    for xmb, ymb in tqdm(text_stream(text, hps.nbatch, hps.nsteps),
                         total=len(text)//(hps.nbatch*hps.nsteps),
                         ncols=125, leave=False):
        cost, smb = sess.run(
            [ema_loss, ema_states],
            {X:xmb, S:smb, Y:ymb}
        )
        costs.append(cost)
    nats = float(np.mean(costs))
    bpc = nats/np.log(2)
    return bpc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--lstm_type',      type=str,   default='scottbrain', choices=['lstm','scottbrain','rnn'])
    parser.add_argument('--nembd',          type=int,   default=64)
    parser.add_argument('--nhidden',        type=int,   default=1120) # 8192
    parser.add_argument('--nproj_in',       type=int,   default=2, help='Sparse input  projection size or stride') # int(round(np.sqrt(blocks)))
    parser.add_argument('--nproj_out',      type=int,   default=2, help='Sparse output projection size or stride') # int(round(np.sqrt(blocks)))
    parser.add_argument('--nsteps',         type=int,   default=64)
    parser.add_argument('--isteps',         type=int,   default=5)
    parser.add_argument('--lsteps',         type=int,   default=1)
    parser.add_argument('--share_isteps',   type=int,   default=0)
    parser.add_argument('--share_masks',    type=int,   default=0)
    parser.add_argument('--block_size',     type=int,   default=32, choices=[8,16,32])
    parser.add_argument('--axis',           type=int,   default=0, choices=[0,1])
    parser.add_argument('--sparsity',       type=str,   default='dense', help='dense | ba_X | bae_X_X')
    parser.add_argument('--dropout',        type=float, default=0.0, help='Whether to add dropout to both internal steps and updates. 0.2 seems to be a good value.')
    parser.add_argument('--dropout_input',  type=int,   default=0, help='Whether to use input dropout.')
    parser.add_argument('--dtype',          type=int,   default=32)
    parser.add_argument('--dx_dtype',       type=int,   default=32)
    parser.add_argument('--dw_dtype',       type=int,   default=32)

    # optimization hyper-parameters
    parser.add_argument('--nepochs',        type=int,   default=70)
    parser.add_argument('--batch_size',     type=int,   default=128, help='Per-GPU batch size')
    parser.add_argument('--ngpu',           type=int,   default=4)
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam_old', 'adam', 'adamax'])
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--lr_warmup_epochs',type=int,  default=5)
    parser.add_argument('--beta2',          type=float, default=.999, help='Adam hyperparameter')
    parser.add_argument('--gamma',          type=float, default=0., help='Adam hyperparameter')
    parser.add_argument('--recompute',      type=int,   default=0, help='Memory efficient backprop: Should be 0 or greater than 1')
    parser.add_argument('--x_group_size',   type=int,   default=16, help='Concat small input and output projection matmuls together')
    parser.add_argument('--forget_bias',    type=float, default=1., help='Forget gate bias')

    # other hyper-parameters
    parser.add_argument('--name',       type=str,   default='', help='experiment label')
    parser.add_argument('--logdir',     type=str, default='logs')
    parser.add_argument('--save_path',  type=str, default='params/params.jl')
    parser.add_argument('--data_file',  type=str, default='/home/scott/datasets/wiki3')
    parser.add_argument('--profile',    type=int, default=0)
    parser.add_argument('--restore',    type=str, default="")
    parser.add_argument('--debug',      type=int, default=0)
    parser.add_argument('--tiny',       type=int, default=0, help='Whether to use tiny dataset')

    dtype_map = {
        7 : tf.bfloat16,
        16: tf.float16,
        32: tf.float32
    }

    args = parser.parse_args()

    args.node     = platform.node()
    args.dtype    = dtype_map[args.dtype]
    args.dx_dtype = dtype_map[args.dx_dtype]
    args.dw_dtype = dtype_map[args.dw_dtype]
    args.nbatch   = args.batch_size * args.ngpu

    # axis 1 not memory efficient with small block sizes so not implemeted yet
    if args.block_size < 32:
        assert args.axis == 0

    # sparse projection not supported on axis 1 yet.. always use full project for dense
    if args.axis == 1 or args.sparsity == "dense":
        args.nproj_in  = args.nhidden
        args.nproj_out = args.nhidden

    if args.sparsity == "dense" or args.share_isteps:
        args.share_masks = True

    args.nproj_in  = min(args.nproj_in,  args.nhidden)
    args.nproj_out = min(args.nproj_out, args.nhidden)

    assert args.recompute == 0 or args.recompute > 1
    if args.recompute > 0:
        # these need to be the same if recompute is enabled
        args.x_group_size = args.recompute
    assert args.x_group_size > 0


    if args.data_file[-5:] == "text8":
        trX, vaX, teX = text8(path=args.data_file)
        text_stream   = text8_stream
        args.nvocab   = 27
    else:
        trX, vaX, teX = wiki3(path=args.data_file)
        text_stream   = wiki3_stream
        args.nvocab   = 256

    hps = HParams(args)

    hps.state_shape = (2, hps.nhidden, hps.nbatch) if hps.axis == 0 else (2, hps.nbatch, hps.nhidden)

    if hps.tiny == 1:
        vaX = trX[1000000:1100000]
        teX = trX[1100000:1200000]
        trX = trX[      0:1000000]

    ntrain = len(trX)
    nval   = len(vaX)
    ntest  = len(teX)

    hps.its_per_epoch = (ntrain-1)//(hps.nbatch*hps.nsteps)
    print("Number of iterations per epoch:", hps.its_per_epoch)

    X = tf.placeholder(tf.int32, [ hps.nsteps, hps.nbatch ])
    Y = tf.placeholder(tf.int32, [ hps.nsteps, hps.nbatch ])
    S = tf.placeholder(tf.float32, hps.state_shape)

    # Create model
    train, loss, states = model(X, S, Y, hps, train=True)
    ema = tf.train.ExponentialMovingAverage(decay=args.beta2)
    avg_params = ema.apply(tf.trainable_variables())
    train = tf.group(train, avg_params)

    if not hps.profile:
        _, ema_loss, ema_states = model(X, S, Y, hps, train=False, ema=ema)

    # Logging
    timestamp = time.strftime('r%Y_%m_%d_%H_%M_%S')
    log_file  = os.path.join(hps.logdir, 'lm', timestamp, "log.txt")
    json_file = os.path.join(hps.logdir, 'lm', timestamp, "json.txt")
    if os.path.exists(log_file):
        # avoid 2 jobs sharing log (quick and dirty fix)
        print(log_file, "already exists, exiting.")
        exit()

    make_path(log_file)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_file, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler()) # Print logs to stderr as well

    hps.num_params = str(num_trainable_params("model0"))
    print_trainable_params("model0")

    json_header = {}
    for key in sorted(hps.__dict__.keys()):
        if type(hps.__dict__[key]) in (str, int, float, type, tf.DType):
            logging.info(str(key) + ': ' + str(hps.__dict__[key]))
            json_header[str(key)] = str(hps.__dict__[key])

    json = JsonLogger(json_file, **json_header)

    # config = tf.ConfigProto(#allow_soft_placement=True,
    #                         intra_op_parallelism_threads=hps.ngpu,
    #                         inter_op_parallelism_threads=hps.ngpu)
    #config.gpu_options.allow_growth = True

    # run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()

    if not args.restore:
        hps.epoch   = 0
        hps.updates = 0
        hps.chars   = 0
        hps.seconds = 0.

    #numerics = 0 #tf.add_check_numerics_ops()
    with tf.Session() as sess: #mu.capture_stderr() as stderr, #config=config

        # We use numpy to init a lot of variables.
        # Rather than use the variable initializers we use placeholders.
        # This trims down the graph_def significantly (keeping it well under 2GB)
        sess.run(tf.global_variables_initializer(), feed_dict=hps.feed_dict)

        # free up memory
        hps.finish_init()

        for i in range(hps.epoch, hps.nepochs):

            smb = np.zeros(hps.state_shape, dtype=np.float32)

            epoch_start = time.time()
            epoch_chars = 0
            it = 0
            for xmb, ymb in tqdm(text_stream(trX, hps.nbatch, hps.nsteps), #, maxbatches=64
                                 total=hps.its_per_epoch,
                                 ncols=125, leave=False):

                if hps.debug and it % hps.debug == 0:
                    smb, _, loss_ = sess.run(
                        [states, train, loss],
                        {X: xmb, S: smb, Y: ymb},
                    )
                    print(it, loss_, loss_/np.log(2))
                    #print("state:", smb)
                else:
                    smb, _ = sess.run(
                        [states, train], #, numerics
                        {X: xmb, S: smb, Y: ymb},
                        #options=run_options, run_metadata=run_metadata
                    )

                # np.savetxt("../test.txt", test, fmt='%6.3f') #,  fmt='%6.3f'
                # exit()

                # mu.print_memory_timeline(stderr, gpu_only=True, ignore_less_than_bytes=1000)
                # exit()

                hps.updates += 1
                hps.chars   += xmb.size
                epoch_chars += xmb.size

                if hps.profile and hps.updates > hps.profile:
                    exit()
                    # tl = timeline.Timeline(run_metadata.step_stats)
                    # ctf = tl.generate_chrome_trace_format()
                    # with open('timeline.json', 'w') as f:
                    #     f.write(ctf)

                it += 1

            epoch_time = time.time() - epoch_start

            hps.epoch   += 1
            hps.seconds += epoch_time

            #hps.save(sess, ema)

            train_bpc = score(trX[:nval], hps)
            valid_bpc = score(vaX, hps)
            test_bpc  = score(teX, hps)
            cps       = epoch_chars // epoch_time
            #print('=' * 125)
            logging.info("nepochs: %3d, train_bpc: %0.6f, valid_bpc: %0.6f, test_bpc: %0.6f, nupdates: %7d, cps: %7d, nseconds: %6d" % (
                          hps.epoch,    train_bpc,        valid_bpc,        test_bpc,        hps.updates,   cps,      hps.seconds ))
            #print('=' * 125)

            json.log(epoch=hps.epoch, train_bpc=train_bpc, valid_bpc=valid_bpc, test_bpc=test_bpc, updates=hps.updates, cps=cps, seconds=hps.seconds)


  # 6.61it/s] 100 2.23159
  # 6.59it/s] 200 2.09856
  # 6.56it/s] 300 2.0249
  # 6.48it/s] 400 1.89606
  # 6.50it/s] 500 1.92836
  # 6.48it/s] 600 1.84487
  # 6.44it/s] 700 1.80276
  # 6.45it/s] 800 1.80252
  # 6.46it/s] 900 1.7833
  # 6.40it/s]1000 1.7205


  # 6.89it/s]100 2.39161
  # 6.83it/s]200 2.19594
  # 6.79it/s]300 2.09597
  # 6.78it/s]400 1.98943
  # 6.82it/s]500 1.93037
  # 6.79it/s]600 1.89484
  # 6.77it/s]700 1.88045


  # 6.88it/s]100 2.34194
  # 6.89it/s]200 2.19156
  # 6.84it/s]300 2.08489
  # 6.82it/s]400 1.99491
  # 6.83it/s]500 1.91741
  # 6.81it/s]600 1.88355
  # 6.80it/s]700 1.85657
  # 6.76it/s]800 1.85184
  # 6.75it/s]900 1.82508

  # 6.74it/s]100 2.32168
  # 6.73it/s]200 2.17876
  # 6.70it/s]300 2.10247
  # 6.70it/s]400 2.00016
  # 6.66it/s]500 1.94257
  # 6.64it/s]600 1.90198
  # 6.68it/s]700 1.86387
  # 6.62it/s]800 1.88786
  # 6.62it/s]900 1.85535
