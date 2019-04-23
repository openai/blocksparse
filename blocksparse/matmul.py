
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from tensorflow.python.framework import ops
from tensorflow.python.ops.init_ops import Initializer
from blocksparse.utils import _op_module, z_order_2d, ceil_div, scalar_constant
import blocksparse.ewops as ew


blocksparse_matmul        = _op_module.blocksparse_matmul
blocksparse_matmul_dx     = _op_module.blocksparse_matmul_dx
blocksparse_matmul_dw     = _op_module.blocksparse_matmul_dw
blocksparse_matmul_dwa    = _op_module.blocksparse_matmul_dwa
blocksparse_matmul_dg     = _op_module.blocksparse_matmul_dg
blocksparse_reduced_dw    = _op_module.blocksparse_reduced_dw
l2_normalize_ck           = _op_module.l2_normalize_ck
l2_normalize_grad_ck      = _op_module.l2_normalize_grad_ck
l2_normalize_gain_ck      = _op_module.l2_normalize_gain_ck
l2_normalize_gain_grad_ck = _op_module.l2_normalize_gain_grad_ck

identity_init = _op_module.blocksparse_matmul_identity_init

# save a bit of gpu memory by only creating one copy of identical constant lookup tables
g_lookup_cache = dict()
g_lut_idx = 0

def get_constant(lut, name):
    global g_lookup_cache
    global g_lut_idx

    default_graph = tf.get_default_graph()
    if name not in g_lookup_cache:
        g_lookup_cache[name] = list()
    for np_entry, tf_entry in g_lookup_cache[name]:
        if np_entry.dtype == lut.dtype and np_entry.shape == lut.shape and tf_entry.graph is default_graph:
            if np.abs(np_entry.astype(np.int64) - lut.astype(np.int64)).sum() == 0:
                # found an exact match
                return tf_entry

    #print(name, lut.size)
    #tf_entry = tf.constant(lut, name=name+"_lut")
    with tf.control_dependencies(None):
        tf_entry = tf.get_variable(f"{name}_lut_{g_lut_idx}", initializer=lut.view(np.int64), trainable=False)
    g_lut_idx += 1

    g_lookup_cache[name].append( (lut, tf_entry) )
    return tf_entry

class IdentityInit(Initializer):

  def __init__(self, lut, CB, KB, blocks, bsize, scale=1.0):
    self.lut = lut
    self.CB = CB
    self.KB = KB
    self.blocks = blocks
    self.bsize  = bsize
    self.scale  = scale

  def __call__(self, shape, dtype=None, partition_info=None):
    assert shape[0] == self.blocks
    #lut = get_constant(self.lut, name="updat")
    with tf.control_dependencies(None):
        lut = tf.constant(self.lut, name="identity_lut")
        return identity_init(lut, CB=self.CB, KB=self.KB, blocks=self.blocks, bsize=self.bsize, scale=self.scale)

SEG_MAX = (1<<63)-1

class BlocksparseMatMul(object):

    def __getstate__(self):
        return (self.layout, self.bsize, self.axis, self.z_order, self.name)

    def __setstate__(self, state):
        self.__init__(*state)

    def __init__(self, layout, block_size=32, feature_axis=0, z_order=True, name=None):

        if (feature_axis == 0 and block_size in (8,16,32)) or \
           (feature_axis == 1 and block_size in (32,64)):
           self.axis   = feature_axis
           self.bsize  = block_size
        else:
            raise ValueError("Unsupported block size with this feature axis")

        assert len(layout.shape) == 2
        CB, KB = layout.shape

        group_sizes = layout.sum(axis=0) # assume symetrical transpose
        max_group = group_sizes.max()
        min_group = group_sizes[np.nonzero(group_sizes)].min()
        if max_group / min_group > 2.0:
            segment_size = max(ceil_div(max_group,4), min_group*2)
        else:
            segment_size = SEG_MAX # not worth segmenting
        #print(max_group, min_group, segment_size, KB)
        #segment_size = SEG_MAX

        # don't creat any segments smaller than this
        seg_min = max(ceil_div(segment_size, 4), 4)

        # segment_size = seg_min = 2

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
        if z_order:
            blk = 0
            for _, i in sorted( [ (z_order_2d(cs[i], ks[i]), i) for i in range(blocks) ] ):
                vs[i] = blk
                updat_list.append((cs[i], ks[i]))
                blk += 1
        else:
            # row contiguous
            updat_list = list( zip(cs, ks) )
            vs = list(range(blocks))
            # cs = [b[0] for b in updat_list]
            # ks = [b[1] for b in updat_list]

        self.updat_list = updat_list
        self.updat_lut  = np.array(updat_list, dtype=np.int32)

        fsetup = self.xprop_lut(KB, cs, ks, vs, idx,  segment_size, seg_min)
        bsetup = self.xprop_lut(CB, ks, cs, vs, idxT, segment_size, seg_min)

        self.fprop_list, self.fprop_lut, self.l2_lut, self.fprop_shared, self.l2_shared, self.fprop_segments, self.fprop_locks = fsetup
        self.bprop_list, self.bprop_lut,           _, self.bprop_shared,              _, self.bprop_segments, self.bprop_locks = bsetup

        if name is None:
            name = "BlocksparseMatMul"

        self.z_order = z_order
        self.name    = name
        self.flops   = blocks * block_size * block_size * 2
        self.blocks  = blocks
        self.w_shape = (blocks, block_size, block_size)
        self.g_shape = (blocks,)
        self.count   = 0

        self.CB = CB
        self.KB = KB
        self.C  = CB * block_size
        self.K  = KB * block_size

        self.sparsity = round(float(blocks) / float(CB * KB), 3)

        # save boolean version for serialization purposes, TODO save csr version
        self.layout = layout > 0


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
                segs.append( (k, []) )
                cols.append( (k, []) )
                #else:
                #    raise ValueError("sparsity mask has empty mappings.  Not yet supported with feature_axis=0")

        #segs.sort(key=lambda x: len(x[1]), reverse=True)

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
        l2_siz = offset + len(vs)
        # we use int64 views into the lut for tf compatibility reasons..
        if l2_siz & 1:
            l2_siz += 1
        l2_lut = np.zeros(l2_siz, dtype=np.int32)
        l2_max = 0
        for i, (k, lut) in enumerate(cols):
            # build the lut header: int offset, lut_size, K
            l2_lut[i*4:(i+1)*4] = offset, len(lut), k, 0
            l2_max = max(l2_max, len(lut))
            for entry in lut:
                l2_lut[offset] = entry[1]
                offset += 1

        return cols, xp_lut, l2_lut, xp_max*8, l2_max*4, len(segs), locks

    def prune(self, param, gate):
        new_blocks = np.sum(gate != 0.0)
        if new_blocks != self.blocks:
            new_param  = np.empty((new_blocks, self.bsize, self.bsize), dtype=param.dtype)
            new_w      = 0
            layout     = self.layout
            for w, (c, k) in enumerate(self.updat_list):
                if gate[w] == 0.0:
                    layout[c,k] = False
                else:
                    new_param[new_w,:,:] = param[w,:,:]
                    new_w += 1
        else:
            new_param = param

        sparsity = round(100 * float(new_blocks) / float(self.CB * self.KB), 1)

        print("prune: ", self.blocks, new_blocks, sparsity)
        return new_param, np.ones((new_blocks,), dtype=gate.dtype)

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

    def identity_init(self, scale=1.0):

        return IdentityInit(self.updat_lut, self.CB, self.KB, self.blocks, self.bsize, scale=scale)

        # def _initializer(shape, dtype=np.float32, partition_info=None):
        #     print("%s identity_init sparsity(%.2f)" % (self.name, self.sparsity))
        #     W = np.zeros(self.w_shape, dtype=dtype)
        #     for w in range(self.blocks):
        #         cb, kb = self.updat_list[w]
        #         if (cb % self.KB) == (kb % self.CB):
        #             W[w] = np.eye(self.bsize, dtype=dtype)
        #     return W
        # return _initializer

    def checker_init(self):
        def _initializer(shape, dtype=np.float32, partition_info=None):
            gate = np.empty(self.blocks, dtype=dtype)
            for w, (c, k) in enumerate(self.updat_list):
                gate[w] = (c & 1) ^ (k & 1) ^ 1
            return gate
        return _initializer

# grid = []
# for c in range(5):
#     row = []
#     for k in range(5):
#         row.append((c & 1) ^ (k & 1) ^ 1)
#     grid.append(row)

# for row in grid:
#     print(row)

    def fprop_test(self, I, W, gate=None):
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
                if gate is None:
                    for c, w in lut:
                        O[k,:,:] += np.dot( W[w,:,:].T, I[c,:,:] ) # CK.T x CN = KN
                else:
                    for c, w in lut:
                        if gate[w] != 0.0:
                            O[k,:,:] += np.dot( W[w,:,:].T, I[c,:,:] ) * gate[w] # CK.T x CN = KN

            return O.reshape(-1, N)

    def bprop_test(self, E, W, gate=None):
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
                if gate is None:
                    for k, w in lut:
                        B[c,:,:] += np.dot( W[w,:,:], E[k,:,:] ) # CK x KN = CN
                else:
                    for k, w in lut:
                        if gate[w] != 0.0:
                            B[c,:,:] += np.dot( W[w,:,:], E[k,:,:] ) * gate[w] # CK x KN = CN

            return B.reshape(-1, N)

    def updat_test(self, I, E, gate=None, dw_gated=False):
        U = np.zeros(self.w_shape)
        bsize = self.bsize
        if self.axis:
            I = I.reshape((-1, self.CB, bsize))
            E = E.reshape((-1, self.KB, bsize))
            for w, (c, k) in enumerate(self.updat_list):
                U[w,:,:] = np.dot( I[:,c,:].T, E[:,k,:] ) # NC.T x NK = CK
        else:
            I = I.reshape((self.CB, bsize, -1))
            E = E.reshape((self.KB, bsize, -1))
            if not dw_gated or gate is None:
                for w, (c, k) in enumerate(self.updat_list):
                    U[w,:,:] = np.dot( I[c,:,:], E[k,:,:].T ) # CN x KN.T = CK
            else:
                for w, (c, k) in enumerate(self.updat_list):
                    if gate[w] != 0.0:
                        U[w,:,:] = np.dot( I[c,:,:], E[k,:,:].T ) * gate[w] # CN x KN.T = CK
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

    def l2_normalize(self, W, gain=None, epsilon=1e-12, dtype=tf.float32):

        l2_lut = get_constant(self.l2_lut, name="l2")

        if gain is None:
            W, _ = l2_normalize_ck(W, l2_lut, TY=dtype, epsilon=epsilon, K=self.K, shared=self.l2_shared, bsize=self.bsize )
        else:
            W, _ = l2_normalize_gain_ck(W, gain, l2_lut, TY=dtype, epsilon=epsilon, K=self.K, shared=self.l2_shared, bsize=self.bsize )
        return W

    def matmul(self, I, W, gate=None, gate_grad=False, dw_gated=False, name=None, bench=0):
        return self.__call__(I, W, gate=gate, gate_grad=gate_grad, dw_gated=dw_gated, name=name, bench=bench)

    def __call__(self, I, W, gate=None, gate_grad=False, dw_gated=False, name=None, bench=0):

        if name is None:
            name = self.name + ("_%06d" % self.count)
        self.count += 1

        if gate is None:
            gate = []
        else:
            gate = [gate]
            #assert self.bsize == 8 and self.axis == 0, "blocksparse gating only implemented for block_size 8 on axis 0"

        fprop_lut = get_constant(self.fprop_lut, name="fprop")
        bprop_lut = get_constant(self.bprop_lut, name="bprop")
        updat_lut = get_constant(self.updat_lut, name="updat")

        O, _ = blocksparse_matmul(
            I, W, fprop_lut, bprop_lut, updat_lut, gate,
            gated_dw=bool(dw_gated), gate_grad=bool(gate_grad),
            blocks=self.blocks, bsize=self.bsize, axis=self.axis, C=self.C, K=self.K,
            segments=self.fprop_segments, segments_dx=self.bprop_segments,
            locks=self.fprop_locks, locks_dx=self.bprop_locks,
            shared=self.fprop_shared, shared_dx=self.bprop_shared, bench=bench, name=name
        )
        #print(O.op.name, O.op.device)
        return O

@ops.RegisterGradient("BlocksparseMatmul")
def blocksparse_matmul_grad(op, dy, temp):

    blocks    = op.get_attr("blocks")
    bsize     = op.get_attr("bsize")
    axis      = op.get_attr("axis")
    C         = op.get_attr("C")
    K         = op.get_attr("K")
    segments  = op.get_attr("segments_dx")
    shared    = op.get_attr("shared_dx")
    locks     = op.get_attr("locks_dx")
    gated_dw  = op.get_attr("gated_dw")
    gate_grad = op.get_attr("gate_grad")
    bench     = op.get_attr("bench")
    x         = op.inputs[0]
    w         = op.inputs[1]
    lut_dx    = op.inputs[3]
    lut_dw    = op.inputs[4]
    gate      = [op.inputs[5]] if len(op.inputs) > 5 else []
    name      = op.name.split('/')[-1]

    dx, _ = blocksparse_matmul_dx(
        dy, w, lut_dx, gate, gated_dw=gated_dw, gate_grad=gate_grad,
        blocks=blocks, bsize=bsize, axis=axis, C=K, K=C, # swap C,K
        segments=segments, locks=locks, shared=shared,
        bench=bench, name=name+"_bprop")

    dw = blocksparse_matmul_dw(
        [x], [dy], lut_dw, gate, gated_dw=gated_dw, gate_grad=gate_grad,
        blocks=blocks, bsize=bsize, axis=axis, C=C, K=K,
        bench=bench, name=name+"_updat")

    # print(dx.op.name, dx.op.device)
    # print(dw.op.name, dw.op.device)

    if len(gate) == 0:
        return (dx, dw, None, None, None)
    elif gate_grad:
        dw, dg = blocksparse_matmul_dg(dw, w, gate[0])
    else:
        dg = None

    return (dx, dw, None, None, None, dg)


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

def block_reduced_full_dw(param_grad, scale=1.0, norm="max", group_size=8):

    # max(abs()) or l2_norm()
    norm  = 0 if norm.lower() == "max" else 1
    # host side scalar, if zero will cause compute for this op to be skipped.
    scale = scalar_constant(scale, dtype=tf.float32)

    assert group_size <= 8

    # backward walk param grad to find BlocksparseMatmulDW ops
    # this should only hit BlocksparseMatmulDWs, BlocksparseMatmulDGs, AddNs or FloatCasts
    ops = get_parents(param_grad, "BlocksparseMatmulDW")
    if len(ops) < 1:
        raise ValueError("BlocksparseMatmulDW op not found")

    # this sorting is dependent on the op names being correctly ordered.
    ops.sort(key=lambda op: op.name.split('/')[-1], reverse=True)

    # use the parent scope for the new ops
    scope = ops[-1].name.split('/')
    scope = '/'.join(scope[0:-1])

    # we're going to be using absolute names, so clear name_scope
    with tf.name_scope(None):
        dw_full = None
        offset  = 0
        while offset < len(ops):

            xs = [op.inputs[0] for op in ops[offset:offset+group_size] ]
            gs = [op.inputs[1] for op in ops[offset:offset+group_size] ]

            # Get the corresponding activation grad op for the last param grad op in the group
            bprop = None
            for consumer in gs[-1].consumers():
                if consumer.type == "BlocksparseMatmulDX":
                    bprop = consumer
                    break
            assert bprop is not None

            # get attributes of first op in group
            up    = ops[offset]
            bsize = up.get_attr("bsize")
            axis  = up.get_attr("axis")
            name  = "%s/block_reduced_full_dw_%03d" % (scope, offset)
            dw_full = [] if dw_full is None else [dw_full]

            dw_full, _, _ = blocksparse_reduced_dw(xs, gs, scale, dw_full, bsize=bsize, norm=norm, axis=axis, name=name)

            # force the dw op before any more time steps are processed
            bprop._add_control_input(dw_full.op)

            offset += group_size

    return dw_full


def group_param_grads(param_grad, group_size=8):

    assert group_size <= 8

    # backward walk param grad to find BlocksparseMatmulDW ops
    # this should only hit BlocksparseMatmulDWs, BlocksparseMatmulDGs, AddNs or FloatCasts
    ops = get_parents(param_grad, "BlocksparseMatmulDW")

    if len(ops) <= 1:
        return param_grad

    # this sorting is dependent on the op names being correctly ordered.
    ops.sort(key=lambda op: op.name.split('/')[-1], reverse=True)
    # for x in ops:
    #     print(x.name)
    # print("")
    # exit()
    segment_size = len(ops)
    if ops[0].get_attr("gate_grad") and len(ops[0].inputs) == 4:
        gate_count = dict()
        max_count  = 0
        for op in ops:
            gate  = op.inputs[3]
            count = gate_count.get(gate, 0) + 1
            gate_count[gate] = count
            max_count = max(max_count, count)
        for count in gate_count.values():
            if count != max_count:
                raise ValueError("Non-uniform gate broadcasting detected.")
        segment_size = max_count
        if  group_size > segment_size:
            group_size = segment_size
        else:
            assert segment_size % group_size == 0
        # nothing to rewrite here.
        if segment_size == 1:
            return param_grad

    # use the parent scope for the new ops
    scope = ops[-1].name.split('/')
    scope = '/'.join(scope[0:-1])

    # we're going to be using absolute names, so clear name_scope
    with tf.name_scope(None):
        dw  = None
        dws = list()
        offset  = 0
        seg_cnt = 0
        while offset < len(ops):

            xs = [op.inputs[0] for op in ops[offset:offset+group_size] ]
            gs = [op.inputs[1] for op in ops[offset:offset+group_size] ]

            # Get the corresponding activation grad op for the last param grad op in the group
            bprop = None
            for consumer in gs[-1].consumers():
                if consumer.type == "BlocksparseMatmulDX":
                    bprop = consumer
                    break
            assert bprop is not None

            # get attributes of first op in group
            up = ops[offset]
            blocks    = up.get_attr("blocks")
            bsize     = up.get_attr("bsize")
            axis      = up.get_attr("axis")
            gated_dw  = up.get_attr("gated_dw")
            gate_grad = up.get_attr("gate_grad")
            C         = up.get_attr("C")
            K         = up.get_attr("K")
            bench     = up.get_attr("bench") // len(xs)
            lut       = up.inputs[2]
            name      = "%s/matmul_concat_updat_%03d" % (scope, offset)
            gate      = [up.inputs[3]] if len(up.inputs) > 3 else []

            # The first op needs to allocate a new dw tensor
            if dw is None:
                dw = blocksparse_matmul_dw(
                    xs, gs, lut, gate, gated_dw=gated_dw,
                    gate_grad=gate_grad, blocks=blocks, bsize=bsize, axis=axis,
                    C=C, K=K, bench=bench, name=name)
            # subsequent ops can just accumulate in place
            else:
                dw = blocksparse_matmul_dwa(
                    xs, gs, lut, dw, gate, gated_dw=gated_dw,
                    gate_grad=gate_grad, blocks=blocks, bsize=bsize, axis=axis,
                    C=C, K=K, bench=bench, name=name)

            # force the dw op before any more time steps are processed
            bprop._add_control_input(dw.op)

            seg_cnt += group_size
            offset  += group_size

            if gate_grad and seg_cnt >= segment_size:
                seg_cnt = 0
                dws.append(dw)
                dw = None

        if gate_grad:
            for i, dw in enumerate(dws):
                # for op in ops[i*group_size:(i+1)*group_size]:
                #     print(op.name)
                # print()
                dw_op  = ops[i*segment_size:(i+1)*segment_size][-1]
                dws[i] = group_dg_grads(dw_op, dw, scope)

            # add up final dw values in groups of 4 for good mix of perforamnce and memory use
            dw = ew.add_n8_op(dws[0:4]) if len(dws) > 1 else dws[0]
            for i in range(4, len(dws), 4):
                dw = ew.add_n8_op(dws[i:i+4] + [dw])

    # splice in these grad op types sitting on top of the param
    if param_grad.op.type in ("Cast", "FloatCast", "L2NormalizeGradCK", "L2NormalizeGainGradCK"):
        param_grad.op._update_input(0, dw)
        dw = param_grad
    elif param_grad.op.type not in ("AddN", "AddN8", "BlocksparseMatmulDW","BlocksparseMatmulDG"):
        raise ValueError("Unexpected grad op type:", param_grad.op.type, param_grad.op.name)

    return dw

def group_dg_grads(bsmm_dw_op, dw, scope):

    # splice the dg + addn ops out of the graph and replace with a single dg op
    # that takes in the final accumulated dw value
    dg_op  = bsmm_dw_op.outputs[0].consumers()[0]
    assert dg_op.type == "BlocksparseMatmulDG"
    dw, dg = blocksparse_matmul_dg(dw, *dg_op.inputs[1:], name=f"{scope}/BlocksparseMatmulDG")

    # splice old add_n op out of graph
    addn_op  = dg_op.outputs[1].consumers()[0]
    addn_ops = list()
    addn_ops.append(addn_op)
    if addn_op.type[0:3] != "Add":
        raise ValueError(f"bad type: {addn_ops[0].type} Cause: this segment does not share a broadcasted gate.")
    elif addn_op.type == "AddN8":
        while True:
            addn_op = addn_op.outputs[0].consumers()[0]
            if addn_op.type == "AddN8":
                addn_ops.append(addn_op)
            else:
                break

    # print(addn_op.name)
    # for i in addn_op.inputs:
    #     print(i.name)
    # print()
    addn = addn_ops[-1].outputs[0]
    dg_consumers = addn.consumers()
    #for op in dg_consumers:

    assert len(dg_consumers) > 0, "raw dg grad not supported"
    #print(addn.name)
    for dg_consumer in dg_consumers:
        found = False
        #print(dg_consumer.name)
        for i, t in enumerate(dg_consumer.inputs):
            #print(i, t.name)
            if t is addn:
                #print(f"splicing dg into: {dg_consumer.name} at {i}")
                dg_consumer._update_input(i, dg)
                found = True
                break
        if not found:
            print(f"splice failed for {dg_consumer.name}")
    return dw


def get_bsmm_dx_ops(param_grad):

    dw_ops = get_parents(param_grad, "BlocksparseMatmulDW")
    dx_ops = list()

    # this sorting is dependent on the op names being correctly ordered.
    dw_ops.sort(key=lambda op: op.name.split('/')[-1], reverse=True)
    for dw_op in dw_ops:
        # Get the corresponding activation grad op
        dx_op = None
        for op in dw_op.inputs[1].consumers():
            if op.type=="BlocksparseMatmulDX":
                dx_op = op
                break
        assert dx_op is not None
        dx_ops.append(dx_op)
    return dx_ops

def get_parents(grad, op_type):
    if grad.op.type == op_type:
        return [grad.op]
    ops  = list()
    wave = set([grad.op])
    while wave:
        new_wave = set()
        for op in wave:
            # print(op.name)
            # for i in op.inputs:
            #     print("   ", i.name)
            # print()
            for op in (t.op for t in op.inputs):
                if op.type == op_type:
                    ops.append(op)
                else:
                    new_wave.add(op)
        wave = new_wave
    return ops

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

    def __getstate__(self):
        return (self.nhidden, self.nproj, self.gather_lut, self.name)

    def __setstate__(self, state):
        self.__init__(state[0], nproj=state[1], gather_lut=state[2], name=state[3])

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
        self.scatter_lut = scatter_lut
        self.nhidden     = nhidden
        self.nproj       = nproj


    def gather(self, x):
        assert x.get_shape()[0].value == self.nhidden
        gather_lut  = get_constant(self.gather_lut,  name="gather")
        scatter_lut = get_constant(self.scatter_lut, name="scatter")
        return gather_scatter_op(x, gather_lut, scatter_lut, C=self.nhidden, K=self.nproj, op=OP_GAT)

    def scatter(self, x):
        assert x.get_shape()[0].value == self.nproj
        gather_lut  = get_constant(self.gather_lut,  name="gather")
        scatter_lut = get_constant(self.scatter_lut, name="scatter")
        return gather_scatter_op(x, scatter_lut, gather_lut, C=self.nproj, K=self.nhidden, op=OP_SCT)

    def scatter_add(self, x, y):
        assert x.get_shape()[0].value == self.nhidden
        assert y.get_shape()[0].value == self.nproj
        gather_lut  = get_constant(self.gather_lut,  name="gather")
        scatter_lut = get_constant(self.scatter_lut, name="scatter")
        return scatter_add_mul_op(x, y, gather_lut, scatter_lut, C=self.nproj, K=self.nhidden, op=OP_ADD)

    def scatter_mul(self, x, y):
        assert x.get_shape()[0].value == self.nhidden
        assert y.get_shape()[0].value == self.nproj
        gather_lut  = get_constant(self.gather_lut,  name="gather")
        scatter_lut = get_constant(self.scatter_lut, name="scatter")
        return scatter_add_mul_op(x, y, gather_lut, scatter_lut, C=self.nproj, K=self.nhidden, op=OP_MUL)


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
