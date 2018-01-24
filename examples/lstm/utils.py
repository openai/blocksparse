import os
import os.path
import string
import json
import numpy as np
import tensorflow as tf
import random

def ceil_div(x, y):
    return -(-x // y)

def text8(path):
    print("opening:", path)
    text = open(path).read()
    tr_text = text[:int(90e6)]
    va_text = text[int(90e6):int(95e6)]
    te_text = text[int(95e6):int(100e6)]
    return tr_text, va_text, te_text

vocab = string.ascii_lowercase+' '
encoder = dict(zip(vocab, range(len(vocab))))
decoder = dict(zip(range(len(vocab)), vocab))

def text8_stream(text, nbatch, nsteps, maxbatches=None):
    nbytes     = len(text)-1
    nperstripe = nbytes//nbatch
    nbatches   = nbytes//(nbatch*nsteps)
    if maxbatches is not None and maxbatches > 0:
        nbatches = min(nbatches, maxbatches)
    xmb        = np.zeros((nbatch, nsteps), dtype=np.int32)
    ymb        = np.zeros((nbatch, nsteps), dtype=np.int32)
    for batch in range(nbatches):
        for n in range(nbatch):
            sidx   = n*nperstripe + batch*nsteps
            xmb[n] = [encoder[byte] for byte in text[sidx:sidx+nsteps]]
            ymb[n] = [encoder[byte] for byte in text[sidx+1:sidx+nsteps+1]]
        # Transpose outputs to get more efficient time step split/concat on axis 0
        yield xmb.T, ymb.T


def text_to_npy(path, nbytes=-1):
    text = np.fromfile(path, dtype=np.uint8, count=nbytes)
    return text

def wiki3(path):
    print("opening:", path)
    tr_text = text_to_npy(os.path.join(path, "wiki.train.raw"))
    va_text = text_to_npy(os.path.join(path, "wiki.valid.raw"))
    te_text = text_to_npy(os.path.join(path, "wiki.test.raw"))
    # the valid/test sets are too small and produce too much variance in the results
    # creat new partitions of 10MB
    text = np.concatenate((tr_text, va_text, te_text))
    te_text = text[:int(10e6)]
    va_text = text[int(10e6):int(20e6)]
    tr_text = text[int(20e6):]

    return tr_text, va_text, te_text

def wiki3_stream(text, nbatch, nsteps, maxbatches=None):
    """
    breaks text into nbatch seperate streams
    yields contiguous nstep sized sequences from each stream until depleted
    """
    nbytes = len(text)-nbatch
    nbatches = nbytes//(nbatch*nsteps)
    if maxbatches is not None:
        nbatches = min([nbatches, maxbatches])
    text = text[:nbatch*nbatches*nsteps+nbatch].reshape(nbatch, -1)
    nperstripe = text.shape[-1]
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    ymb = np.zeros((nbatch, nsteps), dtype=np.int32)
    for start in range(0, nperstripe-1, nsteps):
        # Transpose outputs to get more efficient time step split/concat on axis 0
        yield text[:, start:start+nsteps].T, text[:, start+1:start+nsteps+1].T

# Old stream for amazon
# def text_stream(path, nbatch, nsteps, maxbatches=None):
#     """
#     breaks text into nbatch seperate streams
#     yields contiguous nstep sized sequences from each stream until depleted
#     """
#     text       =  np.fromstring(open(path).read().encode(), dtype=np.uint8)
#     nbytes     = len(text)-1
#     nperstripe = nbytes//nbatch
#     nbatches   = nbytes//(nbatch*nsteps)
#     if maxbatches is not None:
#         nbatches = min(nbatches, maxbatches)
#     xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
#     ymb = np.zeros((nbatch, nsteps), dtype=np.int32)
#     for batch in range(nbatches):
#         for n in range(nbatch):
#             sidx   = n*nperstripe + batch*nsteps
#             xmb[n] = text[sidx:sidx+nsteps]
#             ymb[n] = text[sidx+1:sidx+nsteps+1]
#         # Transpose outputs to get more efficient time step split/concat on axis 0
#         yield xmb.T, ymb.T


class JsonLogger(object):
    def __init__(self, path, **kwargs):
        make_path(path)
        self.path = path
        self.log(**kwargs)

    def log(self, **kwargs):
        file = open(self.path, 'a')
        file.write(json.dumps(kwargs) + '\n')
        file.close()


def ones_initializer(c=1.):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return np.ones(shape, dtype=np.float32) * c
    return _initializer

def zeros_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return np.zeros(shape, dtype=np.float32)
    return _initializer

def normal_initializer(mean=0.0, std=0.02):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return np.random.normal(mean, std, shape).astype(np.float32)
    return _initializer

def ortho_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            print('SHAPE', shape)
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        # return (scale * q[:shape[0], :shape[1]]).astype(np.float32)  # indexing seems pointless here
        return (scale * q).astype(np.float32)

    return _initializer


def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu
    return _assign


def average_grads(tower_grads):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(0, indices)
        values = tf.concat(0, values)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def num_trainable_params(scope):
    return np.sum([np.prod(var.get_shape()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)])


def print_trainable_params(scope):
    print('Variable name, shape, size')
    model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    for var in model:
        print(var.name, var.get_shape(), np.prod(var.get_shape()))
    print('Number of parameters:', np.sum([np.prod(var.get_shape()) for var in model]))


def make_path(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


import math

def constant(p):
    return 1

def linear(p):
    return 1-p

def square(p):
    return (1-p)**2

def sqrt(p):
    return math.sqrt(1-p)

def cube(p):
    return (1-p)**3

def cubert(p):
    return (1-p)**(1/3.)

def fourth(p):
    return (1-p)**4

def fourrt(p):
    return (1-p)**(1/4.)

def cos(p):
    return (math.cos(p*math.pi)+1.)/2.

def sigmoid(p):
    p = p*20-10
    return 1-1/(1+math.exp(-p))

class Scheduler(object):

    def __init__(self, v, nvalues, schedule):
        self.v = v
        self.nvalues = nvalues
        self.schedule = globals()[schedule]

    def value(self, n):
        current_value = self.v*self.schedule(n/self.nvalues)
        return current_value

