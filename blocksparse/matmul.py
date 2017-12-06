
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from tensorflow.python.framework import ops
from tensorflow.python.ops.init_ops import Initializer
import blocksparse.ewops as ew

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))
blocksparse_matmul        = _op_module.blocksparse_matmul
blocksparse_matmul_dx     = _op_module.blocksparse_matmul_dx
blocksparse_matmul_dw     = _op_module.blocksparse_matmul_dw
blocksparse_matmul_dwa    = _op_module.blocksparse_matmul_dwa
l2_normalize_ck           = _op_module.l2_normalize_ck
l2_normalize_grad_ck      = _op_module.l2_normalize_grad_ck
l2_normalize_gain_ck      = _op_module.l2_normalize_gain_ck
l2_normalize_gain_grad_ck = _op_module.l2_normalize_gain_grad_ck

identity_init = _op_module.blocksparse_matmul_identity_init

class IdentityInit(Initializer):

  def __init__(self, lut, CB, KB, blocks, bshift):
    self.lut = lut
    self.CB = CB
    self.KB = KB
    self.blocks = blocks
    self.bshift = bshift

  def __call__(self, shape, dtype=None, partition_info=None):
    assert shape[0] == self.blocks
    return identity_init(self.lut, CB=self.CB, KB=self.KB, blocks=self.blocks, bshift=self.bshift)

SEG_MAX = (1<<63)-1

class BlocksparseMatMul(object):

    def __getinitargs__(self):
        return (self.layout, self.bsize, self.axis, self.name)

    def __init__(self, layout, block_size=32, feature_axis=1, name=None):

        if (feature_axis == 0 and block_size in (8,16,32)) or \
           (feature_axis == 1 and block_size in (32,)):
           self.axis   = feature_axis
           self.bsize  = block_size
           self.bshift = len(bin(block_size)) - 3 # cheap log2(block_size)
        else:
            raise ValueError("Unsupported block size with this feature axis")

        assert len(layout.shape) == 2
        CB, KB = layout.shape

        if self.axis == 1:
            segment_size = SEG_MAX # not yet supported
        else:
            group_sizes = layout.sum(axis=0) # assume symetrical transpose
            max_group = group_sizes.max()
            min_group = group_sizes[np.nonzero(group_sizes)].min()
            if max_group / min_group > 2.0:
                segment_size = max(ceil_div(max_group,4), min_group*2)
            else:
                segment_size = SEG_MAX # not worth segmenting
        #segment_size = SEG_MAX

        # don't creat any segments smaller than this
        seg_min = max(ceil_div(segment_size, 4), 4)

        if layout.dtype != np.int32:
            layout = layout.astype(np.int32)

        # convert to csr for vastly more efficient python iteration on large matrices
        csr = sparse.csr_matrix(layout)
        cs, ks, vs = sparse.find(csr) # ks is in sorted order by default
        blocks = len(vs)
        idx  = list(range(blocks))
        idxT = sorted(idx, key=lambda i: cs[i]) # transpose view

        # morton order (z-order) the blocks for efficient L2 cache utilization across all 3 ops
        updat_list = list()
        blk = 0
        for _, i in sorted( [ (BlocksparseMatMul.morton(cs[i], ks[i]), i) for i in range(blocks) ] ):
            vs[i] = blk
            updat_list.append((cs[i], ks[i]))
            blk += 1
        self.updat_list = updat_list

        updat_lut_np = np.array(updat_list, dtype=np.int32)

        fsetup = self.xprop_lut(KB, cs, ks, vs, idx,  segment_size, seg_min)
        bsetup = self.xprop_lut(CB, ks, cs, vs, idxT, segment_size, seg_min)

        self.fprop_list, fprop_lut_np, l2_lut_np, self.fprop_shared, self.l2_shared, self.fprop_segments, self.fprop_locks = fsetup
        self.bprop_list, bprop_lut_np,         _, self.bprop_shared,              _, self.bprop_segments, self.bprop_locks = bsetup

        if name is None:
            name = "BlocksparseMatMul"

        self.name    = name
        self.flops   = blocks * block_size * block_size * 2
        self.blocks  = blocks
        self.w_shape = (blocks, block_size, block_size)
        self.count   = 0

        self.CB = CB
        self.KB = KB
        self.C  = CB * block_size
        self.K  = KB * block_size

        self.sparsity = float(blocks) / float(CB * KB)

        # save boolean version for serialization purposes, TODO save csr version
        self.layout = layout > 0

        with tf.name_scope(name):
            self.fprop_lut = tf.constant(fprop_lut_np, name="fprop_lut")
            self.bprop_lut = tf.constant(bprop_lut_np, name="bprop_lut")
            self.updat_lut = tf.constant(updat_lut_np, name="updat_lut")

            self.l2_lut = tf.constant(l2_lut_np, name="l2_lut")


    def i_shape(self, N): return (N, self.C) if self.axis else (self.C, N)
    def o_shape(self, N): return (N, self.K) if self.axis else (self.K, N)

    # return the coordinate in the layout that corresponds to a given block id
    def block_coord(self, block): return self.updat_list[block]

    # TODO: write a kernel to do this on the gpu to allow dynamic sparsity
    def xprop_lut(self, KB, cs, ks, vs, idx, max_seg, min_seg):

        locks = 0
        lockids = dict()
        seg  = list()
        segs = list()
        col  = list()
        cols = list()
        kset = set()

        # get a count of channels for each k
        channels = [0 for k in range(KB)]
        for i in idx:
            channels[ks[i]] += 1

        K = ks[idx[0]]
        seg_count = 0
        for i in idx:
            c, k, v = cs[i], ks[i], vs[i]
            kset.add(k)

            # check for new value of k
            if k != K:

                # keep track of unsegmented columns (for l2norm and testing)
                cols.append( (K, col) )
                col = list()

                # append segment for previous K and start a new one
                if len(seg):
                    segs.append( (K, seg) )
                    seg = list()
                    seg_count += 1
                # for more than one segment we need to use spin locks to sync accumulation
                if seg_count > 1:
                    locks += 1
                    lockids[K] = locks
                seg_count = 0
                K = k

            col.append( (c, v) )
            seg.append( (c, v) )

            channels[k] -= 1

            # split columns up into segments, but don't let them be too small for effciency sake
            if len(seg) >= max_seg and channels[k] >= min_seg:
                segs.append( (k, seg) )
                seg = list()
                seg_count += 1

        # append last value of k
        cols.append( (k, col) )
        if len(seg):
            segs.append( (k, seg) )
            seg_count += 1
        if seg_count > 1:
            locks += 1
            lockids[k] = locks

        # add in any empty k blocks at the end
        for k in range(KB):
            if k not in kset:
                # supported by assembly kernels
                if self.axis == 1:
                    segs.append( (k, []) )
                    cols.append( (k, []) )
                else:
                    raise ValueError("sparsity mask has empty mappings.  Not yet supported with feature_axis=0")

        # bsmm lut
        offset = len(segs) * 4
        xp_lut = np.empty(offset + len(vs)*2, dtype=np.int32)
        xp_max = 0
        for i, (k, lut) in enumerate(segs):
            # build the lut header: int2 offset, lut_size, K, lock_id
            xp_lut[i*4:(i+1)*4] = offset//2, len(lut), k, lockids.get(k, 0)
            xp_max = max(xp_max, len(lut))
            for entry in lut:
                xp_lut[offset:offset+2] = entry
                offset += 2

        # l2 norm lut (columns not broken up into segments)
        offset = len(cols) * 4
        l2_lut = np.empty(offset + len(vs), dtype=np.int32)
        l2_max = 0
        for i, (k, lut) in enumerate(cols):
            # build the lut header: int offset, lut_size, K
            l2_lut[i*4:(i+1)*4] = offset, len(lut), k, 0
            l2_max = max(l2_max, len(lut))
            for entry in lut:
                l2_lut[offset] = entry[1]
                offset += 1

        return cols, xp_lut, l2_lut, xp_max*8, l2_max*4, len(segs), locks

    # morton order (z-order)
    @staticmethod
    def morton(x, y):
        answer = 0
        bits = max(len(bin(x)), len(bin(y))) - 2
        for i in range(bits):
            mshifted = 1 << i;
            shift = i
            answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1))
            #print mshifted, shift, answer, bin(answer)
        return answer

    def ortho_init(self):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            W = np.empty(self.w_shape, dtype=dtype)
            bsize = self.bsize
            if self.sparsity < 1.0:
                print("%s ortho_init sparsity(%.2f)" % (self.name, self.sparsity))
                # different block columns are already mostly orthogonal due to sparsity
                # So just make columns within each block of block_size orthogonal
                for k, lut, _ in self.fprop_list:
                    shape = (len(lut) * bsize, bsize)
                    a = np.random.normal(0.0, 1.0, shape).astype(dtype)
                    u, _, v = np.linalg.svd(a, full_matrices=False)
                    if u.shape != shape:
                        u = v
                    for i, (c, w) in enumerate(lut):
                        W[w,:,:] = u[i*bsize:(i+1)*bsize,:]
            else:
                print("%s ortho_init dense" % (self.name,))
                shape = (self.C, self.K)
                a = np.random.normal(0.0, 1.0, shape).astype(dtype)
                u, _, v = np.linalg.svd(a, full_matrices=False)
                if u.shape != shape:
                    u = v
                for w, (c, k) in enumerate(self.updat_list):
                    W[w,:,:] = u[c*bsize:(c+1)*bsize, k*bsize:(k+1)*bsize]

            return W
        return _initializer

    def identity_init(self, gpu=False):
        if gpu:
            return IdentityInit(self.updat_lut, self.CB, self.KB, self.blocks, self.bshift)

        def _initializer(shape, dtype=np.float32, partition_info=None):
            print("%s identity_init sparsity(%.2f)" % (self.name, self.sparsity))
            W = np.zeros(self.w_shape, dtype=dtype)
            for w in range(self.blocks):
                cb, kb = self.updat_list[w]
                if (cb % self.KB) == (kb % self.CB):
                    W[w] = np.eye(self.bsize, dtype=dtype)
            return W
        return _initializer


    def fprop_test(self, I, W):
        bsize = self.bsize
        if self.axis:
            O = np.zeros((I.shape[0], self.KB, bsize))
            I = I.reshape((-1, self.CB, bsize))
            for k, lut in self.fprop_list:
                for c, w in lut:
                    O[:,k,:] += np.dot( I[:,c,:], W[w,:,:] ) # NC x CK = NK
            return O.reshape(I.shape[0], -1)
        else:
            N = I[0].size
            O = np.zeros((self.KB, bsize, N))
            I = I.reshape((self.CB, bsize, N))
            for k, lut in self.fprop_list:
                for c, w in lut:
                    O[k,:,:] += np.dot( W[w,:,:].T, I[c,:,:] ) # CK.T x CN = KN
            return O.reshape(-1, N)

    def bprop_test(self, E, W):
        bsize = self.bsize
        if self.axis:
            B = np.zeros((E.shape[0], self.CB, bsize))
            E = E.reshape((-1, self.KB, bsize))
            for c, lut in self.bprop_list:
                for k, w in lut:
                    B[:,c,:] += np.dot( E[:,k,:], W[w,:,:].T ) # NK x CK.T = NC
            return B.reshape(E.shape[0], -1)
        else:
            N = E[0].size
            B = np.zeros((self.CB, bsize, N))
            E = E.reshape((self.KB, bsize, N))
            for c, lut in self.bprop_list:
                for k, w in lut:
                    B[c,:,:] += np.dot( W[w,:,:], E[k,:,:] ) # CK x KN = CN
            return B.reshape(-1, N)

    def updat_test(self, I, E):
        U = np.empty(self.w_shape)
        bsize = self.bsize
        if self.axis:
            I = I.reshape((-1, self.CB, bsize))
            E = E.reshape((-1, self.KB, bsize))
            for w, (c, k) in enumerate(self.updat_list):
                U[w,:,:] = np.dot( I[:,c,:].T, E[:,k,:] ) # NC.T x NK = CK
        else:
            I = I.reshape((self.CB, bsize, -1))
            E = E.reshape((self.KB, bsize, -1))
            for w, (c, k) in enumerate(self.updat_list):
                U[w,:,:] = np.dot( I[c,:,:], E[k,:,:].T ) # CN x KN.T = CK
        return U

    def l2_normalize_test(self, W, epsilon=1e-12):
        W = W.copy()
        for k, lut in self.fprop_list:
            ws  = [w for c, w in lut]
            W2 = W[ws,:,:].reshape(-1, self.bsize)
            norm = np.sqrt(np.maximum(np.sum(np.square(W2), axis=0, keepdims=True), epsilon))
            for w in ws:
                W[w,:,:] /= norm
        return W

    def l2_normalize_grad_test(self, W, U, epsilon=1e-12):
        for k, lut in self.fprop_list:
            ws = [w for c, w in lut]
            W2 = W[ws,:,:].reshape(-1, self.bsize)
            U2 = U[ws,:,:].reshape(-1, self.bsize)

            sum_sqr_w = np.sum(np.square(W2), axis=0, keepdims=True)
            max_w     = np.maximum(sum_sqr_w, epsilon)
            norm_grad = ( U2 + W2 * (sum_sqr_w >= epsilon) * np.sum(-U2 * W2 / max_w, axis=0, keepdims=True) ) / np.sqrt(max_w)
            norm_grad = norm_grad.reshape(-1, self.bsize, self.bsize)
            for i, w in enumerate(ws):
                U[w,:,:] = norm_grad[i]
        return U

    def l2_normalize(self, W, gain=None, epsilon=1e-12, dtype=np.float32):
        if gain is None:
            W, _ = l2_normalize_ck(W, self.l2_lut, TY=dtype, epsilon=epsilon, K=self.K, shared=self.l2_shared, bsize=self.bsize )
        else:
            W, _ = l2_normalize_gain_ck(W, gain, self.l2_lut, TY=dtype, epsilon=epsilon, K=self.K, shared=self.l2_shared, bsize=self.bsize )
        return W

    def __call__(self, I, W, dw_dtype=tf.float32, name=None, bench=0):

        if name is None:
            name = self.name + ("_%06d" % self.count)
        self.count += 1

        O, _ = blocksparse_matmul(
            I, W, self.fprop_lut, self.bprop_lut, self.updat_lut,
            dtype_y=I.dtype, dtype_dw=dw_dtype,
            blocks=self.blocks, bshift=self.bshift, axis=self.axis, C=self.C, K=self.K,
            segments=self.fprop_segments, segments_dx=self.bprop_segments,
            locks=self.fprop_locks, locks_dx=self.bprop_locks,
            shared=self.fprop_shared, shared_dx=self.bprop_shared, bench=bench, name=name
        )
        #print(O.op.name, O.op.device)
        return O

@ops.RegisterGradient("BlocksparseMatmul")
def blocksparse_matmul_grad(op, dy, temp):

    blocks    = op.get_attr("blocks")
    bshift    = op.get_attr("bshift")
    axis      = op.get_attr("axis")
    C         = op.get_attr("C")
    K         = op.get_attr("K")
    segments  = op.get_attr("segments_dx")
    shared    = op.get_attr("shared_dx")
    locks     = op.get_attr("locks_dx")
    dtype_dw  = op.get_attr("dtype_dw")
    bench     = op.get_attr("bench")
    x         = op.inputs[0]
    w         = op.inputs[1]
    lut_dx    = op.inputs[3]
    lut_dw    = op.inputs[4]
    name      = op.name.split('/')[-1]

    dx, _ = blocksparse_matmul_dx(
        dy, w, lut_dx, dtype_dx=dy.dtype,
        blocks=blocks, bshift=bshift, axis=axis, C=K, K=C, # swap C,K
        segments=segments, locks=locks, shared=shared,
        bench=bench, name=name+"_bprop")

    dw = blocksparse_matmul_dw(
        [x], [dy], lut_dw, dtype_dw=dtype_dw,
        blocks=blocks, bshift=bshift, axis=axis, C=C, K=K,
        bench=bench, name=name+"_updat")

    # print(dx.op.name, dx.op.device)
    # print(dw.op.name, dw.op.device)

    return (dx, dw, None, None, None)


@ops.RegisterGradient("L2NormalizeCK")
def blocksparse_l2_normalize_grad_ck(op, dy, sum_sqr_x):

    epsilon = op.get_attr("epsilon")
    K       = op.get_attr("K")
    shared  = op.get_attr("shared")
    bsize   = op.get_attr("bsize")
    grad_x  = l2_normalize_grad_ck(dy, op.inputs[0], op.outputs[1], op.inputs[1], epsilon=epsilon, K=K, shared=shared, bsize=bsize)

    return (grad_x, None)

@ops.RegisterGradient("L2NormalizeGainCK")
def blocksparse_l2_normalize_grad_ck(op, dy, sum_sqr_x):

    epsilon = op.get_attr("epsilon")
    K       = op.get_attr("K")
    shared  = op.get_attr("shared")
    bsize   = op.get_attr("bsize")
    grad_x, grad_g  = l2_normalize_gain_grad_ck(
        dy, op.inputs[0], op.inputs[1], op.outputs[1], op.inputs[2], epsilon=epsilon, K=K, shared=shared, bsize=bsize)

    return (grad_x, grad_g, None)


# Utils for graph re-writing

def group_param_grads(param_grad, group_size=8, cast32=True):

    assert group_size <= 8

    # backward walk param grad to find BlocksparseMatmulDW ops
    # this should only hit BlocksparseMatmulDWs or AddNs or FloatCasts
    ops = get_parents(param_grad, "BlocksparseMatmulDW")

    # this sorting is dependent on the op names being correctly ordered.
    ops.sort(key=lambda op: op.name.split('/')[-1], reverse=True)
    # for x in ops:
    #     print(x.name)
    # print("")
    # exit()

    # use the parent scope for the new ops
    scope = ops[-1].name.split('/')
    scope = '/'.join(scope[0:-1])

    # we're going to be using absolute names, so clear name_scope
    with tf.name_scope(None):
        offset = 0
        graph  = tf.get_default_graph()
        while offset < len(ops):

            xs = [op.inputs[0] for op in ops[offset:offset+group_size] ]
            gs = [op.inputs[1] for op in ops[offset:offset+group_size] ]

            # Get the corresponding activation grad op for the last param grad op in the group
            bprop = None
            for op in gs[-1].consumers():
                if op.type=="BlocksparseMatmulDX":
                    bprop = op
            assert bprop is not None

            # get attributes of first op in group
            up = ops[offset]
            blocks   = up.get_attr("blocks")
            bshift   = up.get_attr("bshift")
            axis     = up.get_attr("axis")
            dtype_dw = up.get_attr("dtype_dw")
            C        = up.get_attr("C")
            K        = up.get_attr("K")
            bench    = up.get_attr("bench") // len(xs)
            lut      = up.inputs[2]
            name     = "%s/matmul_concat_updat_%03d" % (scope, offset)

            # The first op needs to allocate a new dw tensor
            if offset == 0:
                grad = blocksparse_matmul_dw(
                    xs, gs, lut, dtype_dw=dtype_dw,
                    blocks=blocks, bshift=bshift, axis=axis,
                    C=C, K=K, bench=bench, name=name)
            # subsequent ops can just accumulate in place
            else:
                grad = blocksparse_matmul_dwa(
                    xs, gs, lut, grad,
                    blocks=blocks, bshift=bshift, axis=axis,
                    C=C, K=K, bench=bench, name=name)

            # print(grad.op.name, grad.op.device)

            # force the dw op before any more time steps are processed
            add_control_input(bprop, grad.op)

            #print(grad.op.name)

            offset += group_size

    # get the grad back to float32 if requested
    # TODO: splice the graph instead of this hack
    if cast32 and dtype_dw != tf.float32:
        grad = ew.float_cast(grad, dtype=tf.float32)

    return grad

        # for x in ops:
        #     print(x.name)
        #     print(scope)
        #     print(x.device)
        #     print(x.inputs[3])
        #     print(x.inputs[4])
        #     print("")
        # exit()

def get_parents(grad, op_type):
    ops  = list()
    wave = set([grad.op])
    while wave:
        new_wave = set()
        for op in wave:
            for op in (t.op for t in op.inputs):
                if op.type == op_type:
                    ops.append(op)
                else:
                    new_wave.add(op)
        wave = new_wave
    return ops

def add_control_input(op, control_input):
    op._control_inputs.append(control_input)
    op._recompute_node_def()

def ceil_div(x, y):
    return -(-x // y)

def largest_block(dim):
    for blk in (32,16,8):
        if dim % blk == 0:
            return (blk, dim // blk)
    raise ValueError("dimension not multiple of 8, 16, or 32")

############################## Sparse Projection Ops #####################################

gather_scatter_op   = _op_module.gather_scatter
scatter_add_mul_op  = _op_module.scatter_add_mul
scatter_mul_grad_op = _op_module.scatter_mul_grad

OP_GAT = 0
OP_SCT = 1
OP_ADD = 2
OP_MUL = 3

class SparseProj(object):

    def __getinitargs__(self):
        return (self.nhidden, self.nproj, self.gather_lut, self.name)

    def __init__(self, nhidden, nproj=None, proj_stride=None, block_size=32, gather_lut=None, name=None):

        if gather_lut is None:

            gather_lut = np.arange(nhidden, dtype=np.int32)

            if nproj is not None:

                assert nproj <= nhidden
                np.random.shuffle(gather_lut)
                gather_lut = np.sort(gather_lut[0:nproj])

            elif proj_stride is not None:
                assert proj_stride <= nhidden

                # trim to multiple of block_size
                gather_max = ((nhidden // proj_stride) // block_size) * block_size * proj_stride
                gather_lut = gather_lut[:gather_max:proj_stride].copy()
                nproj      = gather_lut.size
            else:
                raise ValueError("missing nproj, proj_stride or gather_lut")

        if name is None:
            name = "SparseProj"

        # build reverse mapping
        scatter_lut = np.empty(nhidden, dtype=np.int32)
        scatter_lut[:] = -1
        scatter_lut[gather_lut] = np.arange(nproj, dtype=np.int32)

        self.name        = name
        self.gather_lut  = gather_lut
        self.nhidden     = nhidden
        self.nproj       = nproj

        with tf.name_scope(name):
            self.tf_gather  = tf.constant(gather_lut,  name="gather_lut")
            self.tf_scatter = tf.constant(scatter_lut, name="scatter_lut")

    def gather(self, x):
        assert x.get_shape()[0].value == self.nhidden
        return gather_scatter_op(x, self.tf_gather, self.tf_scatter, C=self.nhidden, K=self.nproj, op=OP_GAT)

    def scatter(self, x):
        assert x.get_shape()[0].value == self.nproj
        return gather_scatter_op(x, self.tf_scatter, self.tf_gather, C=self.nproj, K=self.nhidden, op=OP_SCT)

    def scatter_add(self, x, y):
        assert x.get_shape()[0].value == self.nhidden
        assert y.get_shape()[0].value == self.nproj
        return scatter_add_mul_op(x, y, self.tf_gather, self.tf_scatter, C=self.nproj, K=self.nhidden, op=OP_ADD)

    def scatter_mul(self, x, y):
        assert x.get_shape()[0].value == self.nhidden
        assert y.get_shape()[0].value == self.nproj
        return scatter_add_mul_op(x, y, self.tf_gather, self.tf_scatter, C=self.nproj, K=self.nhidden, op=OP_MUL)


@ops.RegisterGradient("GatherScatter")
def gather_scatter_grad(op, dy):
    dx = gather_scatter_op(dy, op.inputs[2], op.inputs[1], C=op.get_attr("K"), K=op.get_attr("C"), op=1-op.get_attr("op"))
    return dx, None, None

@ops.RegisterGradient("ScatterAddMul")
def scatter_add_mul_grad(op, dz):

    if op.get_attr("op") == OP_ADD:
        dx = dz
        dy = gather_scatter_op(dz, op.inputs[2], op.inputs[3], C=op.get_attr("K"), K=op.get_attr("C"), op=OP_GAT)
    else:
        dx, dy = scatter_mul_grad_op(dz, *op.inputs[0:3], C=op.get_attr("C"), K=op.get_attr("K"))

    return dx, dy, None, None


# REGISTER_OP("GatherScatter")
#     .Input("x: T")
#     .Input("gather: int32")
#     .Input("scatter: int32")
#     .Output("y: T")
#     .Attr("T: {half, float, bfloat16}")
#     .Attr("C: int")
#     .Attr("K: int")
#     .Attr("op: int")

# REGISTER_OP("ScatterAddMul")
#     .Input("x: T")
#     .Input("y: T")
#     .Input("gather: int32")
#     .Input("scatter: int32")
#     .Output("z: T")
#     .Attr("T: {half, float, bfloat16}")
#     .Attr("C: int")
#     .Attr("K: int")
#     .Attr("op: int")

# REGISTER_OP("ScatterMulGrad")
#     .Input("dz: T")
#     .Input("x: T")
#     .Input("y: T")
#     .Input("gather: int32")
#     .Output("dx: T")
#     .Output("dy: T")
#     .Attr("T: {half, float, bfloat16}")
#     .Attr("C: int")
#     .Attr("K: int")