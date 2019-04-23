
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from operator import lt
from tensorflow.python.framework import ops

from blocksparse.utils import _op_module, reduce_mul, ceil_div, z_order_3d, magic32u, magic64u

blocksparse_conv_op     = _op_module.blocksparse_conv
blocksparse_deconv_op   = _op_module.blocksparse_deconv

edge_bias_op            = _op_module.edge_bias
edge_bias_grad_op       = _op_module.edge_bias_grad

l2_normalize_kctrs      = _op_module.l2_normalize_kctrs
l2_normalize_cktrs      = _op_module.l2_normalize_cktrs
l2_normalize_grad_kctrs = _op_module.l2_normalize_grad_kctrs
l2_normalize_grad_cktrs = _op_module.l2_normalize_grad_cktrs

l2_normalize_gain_kctrs      = _op_module.l2_normalize_gain_kctrs
l2_normalize_gain_cktrs      = _op_module.l2_normalize_gain_cktrs
l2_normalize_gain_grad_kctrs = _op_module.l2_normalize_gain_grad_kctrs
l2_normalize_gain_grad_cktrs = _op_module.l2_normalize_gain_grad_cktrs


# float_cast_op           = _op_module.float_cast

#convenince wrappers:

# from blocksparse.conv import conv_edge_bias_init

# y = tf.nn.conv2d(x, w, stride_shape, pad, data_format="NHWC")

# edge_bias_op = conv_edge_bias_init(y, x, w, stride_shape, pad, data_format="NHWC")

# eg = tf.get_variable("EG", edge_bias_op.shape, tf.float32, initializer=tf.ones_initializer())
# eb = tf.get_variable("EB", edge_bias_op.shape, tf.float32, initializer=tf.zeros_initializer())

# y = edge_bias_op(y, eg, eb)

def conv_edge_bias_init(y, x, w, strides=None, padding="SAME", data_format="NHWC", dilations=None):
    return ConvEdgeBias(y.shape.as_list(), x.shape.as_list(), w.shape.as_list(), strides, padding, data_format, dilations)


def deconv_edge_bias_init(y, x, w, strides=None, padding="SAME", data_format="NHWC", dilations=None):
    # swap x and y
    return ConvEdgeBias(x.shape.as_list(), y.shape.as_list(), w.shape.as_list(), strides, padding, data_format, dilations, deconv=True)


class ConvEdgeBias(object):

    Cache = dict()

    def __init__(self, y_shape, x_shape, w_shape, strides=None, padding="SAME", data_format="NHWC", dilations=None, deconv=False):

        if data_format in ("NCW","NCHW","NCDHW"):
            self.layout = 0
            sdim = slice(2,None) # NCHW
            #fdim = slice(2,None) # KCRS
            # tf keeps its own format for params and does transpose ops..
            fdim = slice(0,-2) # RSCK
            cdim = 1
        else:
            self.layout = 1
            sdim = slice(1,-1) # NHWC
            fdim = slice(0,-2) # RSCK
            cdim = -1

        C = x_shape[cdim]
        K = y_shape[cdim]
        MPQ = expand_dims(y_shape[sdim])
        DHW = expand_dims(x_shape[sdim])
        TRS = expand_dims(w_shape[fdim])

        strides = (1,1,1) if strides   is None else expand_dims(strides[sdim])
        dilates = (1,1,1) if dilations is None else expand_dims(dilations[sdim])

        if padding.upper() == "VALID":
            padding = (0,0,0)
        else:
            padding = list()
            for S, Q, W, stride, dilate in zip(TRS, MPQ, DHW, strides, dilates):
                # match padding formula used in tensorflow
                padding.append(max((Q - 1) * stride + S - W, 0) // 2)

        if deconv:
            lut_func = bprop_lut
            MPQ, DHW = DHW, MPQ
            C, K     = K, C
        else:
            lut_func = fprop_lut

        key = tuple(tuple(a) for a in (MPQ, DHW, TRS, padding, strides, dilates))

        entry = ConvEdgeBias.Cache.get(key, None)
        if entry is None:

            mpqLut = list()
            fdata  = list(zip(TRS, padding, strides, dilates))
            for i in range(3):
                mpqLut.append( [ lut_func( dim, DHW[i], *fdata[i]) for dim in range(MPQ[i]) ] )

            self._build_edge_lut(MPQ, mpqLut)

            ConvEdgeBias.Cache[key] = (self.edgeBiasMap, self.edgeBiasLut, self.edgeEntries)
        else:
            self.edgeBiasMap, self.edgeBiasLut, self.edgeEntries = entry
            self.edgeBiasDim = len(self.edgeBiasMap)

        self.shape = (self.edgeBiasDim, K) if self.layout else (K, self.edgeBiasDim)


    def _build_edge_lut(self, MPQ, mpqLut):

        # Hash the mpq coordinates on unique edge overlap patterns
        # The hash key is the list of lut indicies where the offset is -1
        PQ = MPQ[1] * MPQ[2]
        Q  = MPQ[2]
        edge_map = dict()
        mLut, pLut, qLut = mpqLut
        for m,p,q in np.ndindex(*MPQ):
            key = list()
            for di, d in enumerate(mLut[m]):
                for hi, h in enumerate(pLut[p]):
                    for wi, w in enumerate(qLut[q]):
                        if any(x == -1 for x in (d,h,w)):
                            key.append((di,hi,wi))
            if len(key):
                key = tuple(key)
                mpqOffset = m*PQ + p*Q + q
                edge_list = edge_map.get(key)
                if edge_list is None:
                    edge_map[key] = [mpqOffset]
                else:
                    edge_list.append(mpqOffset)

        self.edgeBiasDim = len(edge_map)

        if self.edgeBiasDim:
            # so K x len(edge_map) is the size of the bias vector
            # we need a lut of bias index => mpqOffset mappings
            biasHead = list()
            biasData = list()
            biasMap  = sorted(edge_map.values(), key=lambda x: x[0])
            offset   = len(biasMap) * 2
            # the lut contains a header with 2 entries per unique bias: offset, size
            for mpqList in biasMap:
                biasHead.extend((offset, len(mpqList)))
                biasData.extend(mpqList)
                offset += len(mpqList)

            pad4 = 4 - (len(biasData) & 3) if (len(biasData) & 3) else 0
            biasLut = biasHead + biasData + ( [0] * pad4 )
            self.edgeEntries = len(biasData)
            self.edgeBiasMap = biasMap
            self.edgeBiasLut = tf.constant(np.array(biasLut, dtype=np.int32), name="edge_bias_lut")

    def edge_bias_test(self, x, g, b):
        if self.edgeBiasDim:
            if self.layout:
                N = x.shape[0]
                K = x.shape[-1]
                y = np.array(x.reshape(N, -1, K))
                for i in range(self.edgeBiasDim):
                    y[:,self.edgeBiasMap[i],:] = y[:,self.edgeBiasMap[i],:] * g[i,:].reshape(1, 1, K) + b[i, :].reshape(1, 1, K)
                return y.reshape(x.shape)
            else:
                N, K = x.shape[0:2]
                y = np.array(x.reshape(N, K, -1))
                for i in range(self.edgeBiasDim):
                    y[:,:,self.edgeBiasMap[i]] = y[:,:,self.edgeBiasMap[i]] * g[:,i].reshape(1, K, 1) + b[:,i].reshape(1, K, 1)
                return y.reshape(x.shape)
        else:
            return x

    # dx = g * dy
    # dg = sum(dy * x)
    # db = sum(dy)
    def edge_bias_grad_test(self, dy, x, g):
        if self.edgeBiasDim:
            lut = self.edgeBiasMap
            dy_shape = dy.shape
            if self.layout:
                N = dy_shape[0]
                K = dy_shape[-1]
                x  = x.reshape(N, -1, K)
                dy = dy.reshape(N, -1, K)
                dx = np.array(dy)
                dg = np.empty(self.shape, dtype=np.float32)
                db = np.empty(self.shape, dtype=np.float32)
                for i in range(self.edgeBiasDim):
                    dx[:,lut[i],:] *= g[i,:].reshape(1, 1, K)
                    dg[i,:] = (dy[:,lut[i],:] * x[:,lut[i],:]).sum(axis=(0,1))
                    db[i,:] = dy[:,lut[i],:].sum(axis=(0,1))
            else:
                N, K = dy_shape[0:2]
                x  = x.reshape(N, K, -1)
                dy = dy.reshape(N, K, -1)
                dx = np.array(dy)
                dg = np.empty(self.shape, dtype=np.float32)
                db = np.empty(self.shape, dtype=np.float32)
                for i in range(self.edgeBiasDim):
                    dx[:,:,lut[i]] *= g[:,i].reshape(1, K, 1)
                    dg[:,i] = (dy[:,:,lut[i]] * x[:,:,lut[i]]).sum(axis=(0,2))
                    db[:,i] = dy[:,:,lut[i]].sum(axis=(0,2))

            return dx.reshape(dy_shape), dg, db
        else:
            return dy, None, None

    def __call__(self, x, g, b, inference=False, bench=0, name=None):
        if self.edgeBiasDim:
            return edge_bias_op(x, g, b, self.edgeBiasLut, layout=self.layout, entries=self.edgeEntries, inference=inference, bench=bench, name=name)
        return x

@ops.RegisterGradient("EdgeBias")
def edge_bias_grad(op, dy):

    dx, dg, db = edge_bias_grad_op(dy, op.inputs[0], op.inputs[1], op.inputs[3], layout=op.get_attr("layout"), entries=op.get_attr("entries"), bench=op.get_attr("bench"))
    return (dx, dg, db, None)


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
    """
    def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", debug=False, deconv=False):

        # save this so we know the users perfered number of dims (before we pad 1's out to 3 dims)
        self.userTRS = list(TRS)

        # support 1-3 dims (additional dimensions are possible by purely extending this python code)
        for a in (TRS, DHW, MPQ, strides, dilates, padding):
            if type(a) in (tuple, list):
                assert 1 <= len(a) <= 3
        assert len(TRS) == len(DHW)

        # Process the spatial dimensions

        # pad sizes and strides out to 3 dimensions
        TRS = expand_dims(TRS)
        DHW = expand_dims(DHW)
        strides = expand_dims(strides)
        dilates = expand_dims(dilates)
        padding = get_padding(padding, TRS, dilates)

        if MPQ is None:
            MPQ = [ out_dim(*dims) for dims in zip(TRS, DHW, padding, strides, dilates) ]
        else:
            MPQ = expand_dims(MPQ)

        trs = reduce_mul(TRS)
        dhw = reduce_mul(DHW)
        mpq = reduce_mul(MPQ)

        # contruct feature portion of the grid data loaded to each cuda block
        cMax = kMax = sizeF = 0
        overlapC = overlapK = False
        cSet, kSet = set(), set()
        ckLut      = list()
        fpropGridF = list()
        bpropGridF = list()
        updatGridF = list()
        normList   = list()
        blkSizes   = set()

        for listC, listK in BCK:
            offset_C = list()
            for c in listC:
                offset_C.append(c * dhw)
                if c in cSet:
                    overlapC = True
                else:
                    cSet.add(c)

            offset_K = list()
            for k in listK:
                offset_K.append(k * mpq)
                if k in kSet:
                    overlapK = True
                else:
                    kSet.add(k)

            block_C   = len(listC)
            block_K   = len(listK)
            offset_CK = len(ckLut)
            cMax = max(cMax, block_C)
            kMax = max(kMax, block_K)
            CTRS = block_C*trs
            KTRS = block_K*trs
            blkSizes.add((block_K, block_C))

            # fprop: K is the outer product dim
            fpropGridF.append( [ ceil_div(block_K, 32), block_C, block_K, offset_CK, sizeF ] )

            # bprop: C is the outer product dim
            bpropGridF.append( [ ceil_div(block_C, 32), block_C, block_K, offset_CK, sizeF ] )

            # update: K and CTRS are the outer dims (KCRS = KPQ x CHW.T)
            updatGridF.append( [ ceil_div(CTRS, 32), ceil_div(block_K, 32), block_C, block_K, offset_CK, sizeF ] )

            # setup luts for weight norm
            if deconv:
                # for deconv, C and K were swapped coming in, so we need to unswap them
                for c in range(block_C):
                    normList.append((c, KTRS, CTRS, sizeF))
            else:
                for k in range(block_K):
                    normList.append((sizeF + k * CTRS, CTRS))

            # total filter size (and current filter block offset)
            sizeF += block_K * block_C * trs

            ckLut.extend(offset_C)
            ckLut.extend(offset_K)

        ckLut = np.array(ckLut, dtype=np.int32)

        # Assume no missing mappings.
        self.C = len(cSet)
        self.K = len(kSet)
        self.fixed_block_size =  len(blkSizes) == 1

        # Process the spatial component of the grid
        self.mpqLut = list()
        self.dhwLut = list()
        self.mpqSlice = None
        fdata = list(zip(TRS, padding, strides, dilates))
        for i in range(3):
            self.mpqLut.append( [ fprop_lut( x, DHW[i], *fdata[i]) for x in range(MPQ[i]) ] )
            self.dhwLut.append( [ bprop_lut( x, MPQ[i], *fdata[i]) for x in range(DHW[i]) ] )
        mpq_lut = self.spatial_grid(DHW, MPQ, self.mpqLut, mpq, trs)
        dhw_lut = self.spatial_grid(MPQ, DHW, self.dhwLut, dhw, trs)

        # get the super block dimension
        dim_O = mpq_lut.shape[0]
        dim_I = dhw_lut.shape[0]

        # merge the spatial and feature outer product grid info
        fpropGrid = list()
        for dim_K, block_C, block_K, offset_CK, offset_F in fpropGridF:
            for order, idx_MPQ, idx_K in sorted([ (z_order_3d(0,o,k), o,k) for o,k in np.ndindex(dim_O, dim_K) ]):
                # idx_K/idx_MPQ, block_K/block_C, offset_CK, offset_F
                fpropGrid.append( [
                    idx_MPQ + (idx_K   << 16),
                    block_C + (block_K << 16),
                    offset_CK, offset_F ] )

        bpropGrid = list()
        for dim_C, block_C, block_K, offset_CK, offset_F in bpropGridF:
            for order, idx_DHW, idx_C in sorted([ (z_order_3d(0,i,c), i,c) for i,c in np.ndindex(dim_I, dim_C) ]):
                # idx_C/idx_DHW, block_K/block_C, offset_CK, offset_F
                bpropGrid.append( [
                    idx_DHW + (idx_C   << 16),
                    block_C + (block_K << 16),
                    offset_CK, offset_F ] )

        updatGrid = list()
        for dim_CTRS, dim_K, block_C, block_K, offset_CK, offset_F in updatGridF:
            for order, idx_MPQ, idx_K, idx_CTRS in sorted([ (z_order_3d(o,k,c), o,k,c) for o,k,c in np.ndindex(dim_O, dim_K, dim_CTRS) ]):
                # idx_MPQ, idx_CTRS/idx_K, block_C, block_K, offset_CK, offset_F
                updatGrid.append( [
                    idx_MPQ, idx_CTRS + (idx_K << 16),
                    block_C, block_K,
                    offset_CK, offset_F ] )

        fpropGrid = np.array(fpropGrid, dtype=np.int32)
        bpropGrid = np.array(bpropGrid, dtype=np.int32)
        updatGrid = np.array(updatGrid, dtype=np.int32)
        normLut   = np.array(normList,  dtype=np.int32)

        self.fshared = (trs*32 + 32 + ceil_div(cMax,4)*4 + min(kMax,32)) * 4
        self.bshared = (trs*32 + 32 + ceil_div(kMax,4)*4 + min(cMax,32)) * 4

        # flops per image of minibatch
        self.flops  = sizeF * mpq * 2
        self.blocks = len(BCK)
        self.debug  = bool(debug)

        self.BCK = BCK
        self.TRS = TRS
        self.DHW = DHW
        self.MPQ = MPQ
        self.sizeF = sizeF
        self.strides = strides
        self.dilates = dilates
        self.padding = padding

        # For integer division we'd like to do this in a single XMAD sass instruction (plus shift).
        # We need to be inside of 16 bits for this to work.
        # An additional XMAD could be added at a slight performance loss to support larger dimensions.
        # But I'm not sure these larger dimensions are needed in practice.
        cktrsMax   = ceil_div(max(cMax, kMax)*trs, 32) * 32
        cktrsMagic = magic32u(cktrsMax, trs)
        assert cktrsMax < 2**16 and cktrsMagic[0] < 2**16, \
            "Use cuDNN for large single blocks, but email me if you think there is a use case for this: scott@openai.com"

        # kernel params
        self.trs = trs
        self.magic_trs = cktrsMagic
        self.overlapC = overlapC
        self.overlapK = overlapK
        self.normSize = len(normList)

        self.ck_lut     = tf.constant(ckLut,     name="ck_lut")
        self.mpq_lut    = tf.constant(mpq_lut,   name="mpq_lut")
        self.dhw_lut    = tf.constant(dhw_lut,   name="dhw_lut")
        self.fprop_grid = tf.constant(fpropGrid, name="fprop_grid")
        self.bprop_grid = tf.constant(bpropGrid, name="bprop_grid")
        self.updat_grid = tf.constant(updatGrid, name="updat_grid")
        self.norm_lut   = tf.constant(normLut,   name="norm_lut")


    def spatial_grid(self, DHW, MPQ, mpqLut, mpq, trs):

        # Find the most efficient super-block using a tile of size 32
        # For ties then pick the larger tile in the W dim (more contiguous memory access)
        # TODO: allow a mixture of superblock shapes, or maybe odd shapes to get better ulilization
        ulilization = list()
                  # xxxxx    yxxxx    yyxxx   zyyxx
        for sb in ((1,1,32),(1,2,16),(1,4,8),(2,4,4)):
            util = float(mpq) / reduce_mul( [ ceil_div(*dims) for dims in zip(MPQ, sb) ], 32)
            ulilization.append((1.0 - util, 32 - sb[2], sb))
        sb = sorted(ulilization)[0][2]

        # Map the 32 positions in the superblock to MPQ coordinates
        # superblock mask: zyyxx : (1,3,3), yxxxx : (0,1,15)
        sb_mask  = [ x - 1 for x in sb ]
        # superblock cumulative right-shift: zyyxx : (4,2,0), yxxxx : (5,4,0)
        shifts = [ len(bin(x)) - 3 for x in sb ]
        sb_shift = [ shifts[1]+shifts[2], shifts[2], 0 ]

        HW = DHW[1] * DHW[2]
        W  = DHW[2]
        PQ = MPQ[1] * MPQ[2]
        Q  = MPQ[2]

        # Get the dimension in super blocks
        mpqDim = [ ceil_div(MPQ[i], sb[i]) for i in range(3) ]
        mpq_lut = list()

        # Iterate over superblocks to build the lut
        for order, sb_mpq in sorted([ (z_order_3d(*mpq), mpq) for mpq in np.ndindex(*mpqDim) ]):

            lut32 = [ list() for i in range(trs+1) ]
            for i32 in range(32):

                # get the mpq coord for each of the 32 positions in the superblock
                m = sb_mpq[0] * sb[0] + ((i32 >> sb_shift[0]) & sb_mask[0])
                p = sb_mpq[1] * sb[1] + ((i32 >> sb_shift[1]) & sb_mask[1])
                q = sb_mpq[2] * sb[2] + ((i32 >> sb_shift[2]) & sb_mask[2])

                # make sure we didn't fall off the edge
                if all(lt(*mM) for mM in zip((m,p,q), MPQ)):
                    # add in all the input image offsets for each filter position
                    lut = [ d*HW + h*W + w if all(x >= 0 for x in (d,h,w)) else -1
                            for d in mpqLut[0][m]
                            for h in mpqLut[1][p]
                            for w in mpqLut[2][q] ]

                    # add the output image offset
                    lut.append( m*PQ + p*Q + q )
                else:
                    # -1 offsets get zero padded
                    lut = [-1] * (trs+1)

                # transpose lut data so contiguous rows are for 32 mpq coords of the same trs value
                for i in range(trs+1):
                    lut32[i].append(lut[i])

            mpq_lut.append(lut32)

        return np.array(mpq_lut, dtype=np.int32)

    def i_shape(self, N): return [N, self.C] + self.DHW
    def o_shape(self, N): return [N, self.K] + self.MPQ
    def f_shape(self, block=None):
        if block is None:
            if self.fixed_block_size:
                lutC, lutK = self.BCK[0]
                return [self.blocks, len(lutK), len(lutC)] + self.userTRS
            return [self.sizeF,]
        lutC, lutK = self.BCK[block]
        return [len(lutK), len(lutC)] + self.userTRS


    def __call__(self, F, I):
        assert I.get_shape()[1] == self.C
        output = blocksparse_conv_op(
            self.fprop_grid, self.bprop_grid, self.updat_grid,
            self.mpq_lut, self.dhw_lut, self.ck_lut,
            F, I, c_type=I.dtype,
            mode=0, overlapC=self.overlapC, overlapK=self.overlapK,
            C=self.C, K=self.K, DHW=self.DHW, MPQ=self.MPQ, trs=self.trs,
            magic_trs=self.magic_trs[0], shift_trs=self.magic_trs[1],
            dimF=F.get_shape().as_list(), fshare=self.fshared, bshare=self.bshared, debug=self.debug
        )
        return output

    def l2_normalize(self, F, gain=None, epsilon=1e-12, dtype=np.float32):
        if gain is None:
            F, _ = l2_normalize_kctrs(F, self.norm_lut, TY=dtype, epsilon=epsilon, K=self.normSize )
        else:
            assert self.overlapK is False, "no gain support for overlapping output blocks"
            F, _ = l2_normalize_gain_kctrs(F, gain, self.norm_lut, TY=dtype, epsilon=epsilon, K=self.normSize )
        return F

    def collapse_filter(self, F, dtype=None):
        flatF = np.empty(self.sizeF, dtype=dtype)
        offset = 0
        for f in F:
            flatF[offset:offset+f.size] = f.reshape(f.size).astype(dtype)
            offset += f.size
        return flatF

    def init_slices(self):
        if self.mpqSlice is None:
            self.mpqSlice = list()
            self.dhwSlice = list()
            fdata  = list(zip(self.TRS, self.padding, self.strides, self.dilates))
            for i in range(3):
                self.mpqSlice.append( [ fprop_slice(x, self.DHW[i], *fdata[i]) for x in range(self.MPQ[i]) ] )
                self.dhwSlice.append( [ bprop_slice(x, self.MPQ[i], *fdata[i]) for x in range(self.DHW[i]) ] )

    def fprop_test(self, F, I, alpha=1.0):
        self.init_slices()
        N = I.shape[0]
        O = np.zeros([N, self.K] + self.MPQ)
        mSlice, pSlice, qSlice = self.mpqSlice
        for block in range(self.blocks):

            blockF = F[block]
            blockK = blockF.shape[0]
            lutC, lutK = self.BCK[block]

            for m,p,q in np.ndindex(*self.MPQ):
                sliceT, sliceD, _ = mSlice[m]
                sliceR, sliceH, _ = pSlice[p]
                sliceS, sliceW, _ = qSlice[q]

                # KxCTRS
                slicedF = blockF[:,:,sliceT,sliceR,sliceS].reshape((blockK, -1))
                # NxCDHW
                slicedI = I[:,lutC,sliceD,sliceH,sliceW].reshape((N, -1))
                # NxKMPQ
                O[:,lutK,m,p,q] += np.dot( slicedI, slicedF.T ) * alpha

        return O

    def bprop_test(self, F, I, alpha=1.0):
        self.init_slices()
        N = I.shape[0]
        O = np.zeros([N, self.C] + self.DHW)
        dSlice, hSlice, wSlice = self.dhwSlice
        for block in range(self.blocks):

            # KC => CK, invert TRS
            blockF = np.transpose(F[block][:,:,::-1,::-1,::-1], (1,0,2,3,4)).copy()
            blockC = blockF.shape[0]
            lutC, lutK = self.BCK[block]

            for d,h,w in np.ndindex(*self.DHW):
                sliceT, sliceM = dSlice[d]
                sliceR, sliceP = hSlice[h]
                sliceS, sliceQ = wSlice[w]

                # CxKTRS
                slicedF = blockF[:,:,sliceT,sliceR,sliceS].reshape((blockC, -1))
                # NxKMPQ
                slicedI = I[:,lutK,sliceM,sliceP,sliceQ].reshape((N, -1))
                # NxCDHW
                O[:,lutC,d,h,w] += np.dot( slicedI, slicedF.T ) * alpha

        return O

    def updat_test(self, E, I, alpha=1.0, transpose=False):
        self.init_slices()
        U = list()
        N = I.shape[0]
        mSlice, pSlice, qSlice = self.mpqSlice
        for block in range(self.blocks):

            lutC, lutK = self.BCK[block]
            dimF = self.f_shape(block)
            blockU = np.zeros(dimF)
            U.append(blockU)

            for m,p,q in np.ndindex(*self.MPQ):
                sliceT, sliceD, tlen = mSlice[m]
                sliceR, sliceH, rlen = pSlice[p]
                sliceS, sliceW, slen = qSlice[q]

                # NxCDHW
                slicedI = I[:,lutC,sliceD,sliceH,sliceW].reshape(N,-1)
                # NxKMPQ
                slicedE = E[:,lutK,m,p,q]
                # CxKTRS
                blockU[:,:,sliceT,sliceR,sliceS] += np.dot(slicedE.T, slicedI).reshape((dimF[0], dimF[1], tlen, rlen, slen)) * alpha

        return self.collapse_filter(U, dtype=np.float32)

    def l2_normalize_test(self, F, gain=None, epsilon=1e-12):
        normF = list()
        if gain is None:
            for blockF in F:
                norm = np.sqrt(np.maximum(np.sum(np.square(blockF), axis=(1,2,3,4), keepdims=True), epsilon))
                normF.append((blockF / norm))
        else:
            offsetK = 0
            for blockF in F:
                blockK = blockF.shape[0]
                g = gain[offsetK:offsetK+blockK].reshape((blockK,1,1,1,1))
                norm = np.sqrt(np.maximum(np.sum(np.square(blockF), axis=(1,2,3,4), keepdims=True), epsilon))
                normF.append((g * blockF / norm))
                offsetK += blockK
        return self.collapse_filter(normF, dtype=np.float32)

    def l2_normalize_grad_test(self, F, U, gain=None, epsilon=1e-12):
        D = list()
        if gain is None:
            grad_g = None
            for blockF, blockU in zip(F, U):

                sum_sqr_w = np.sum(np.square(blockF), axis=(1,2,3,4), keepdims=True)
                max_w     = np.maximum(sum_sqr_w, epsilon)
                d         = ( blockU + blockF * (sum_sqr_w >= epsilon) * np.sum(-blockU * blockF / max_w, axis=(1,2,3,4), keepdims=True) ) / np.sqrt(max_w)
                D.append(d)

        else:
            grad_g = np.empty(self.K)
            offsetK = 0
            for blockF, blockU in zip(F, U):
                blockK = blockF.shape[0]
                g = gain[offsetK:offsetK+blockK].reshape((blockK,1,1,1,1))

                sum_sqr_w = np.sum(np.square(blockF), axis=(1,2,3,4), keepdims=True)
                max_w     = np.maximum(sum_sqr_w, epsilon)
                norm_w    = 1.0 / np.sqrt(max_w)

                grad_g[offsetK:offsetK+blockK] = np.sum(blockU * blockF * norm_w, axis=(1,2,3,4))

                d = ( blockU * g + blockF * (sum_sqr_w >= epsilon) * np.sum(-blockU * blockF * g / max_w, axis=(1,2,3,4), keepdims=True) ) * norm_w
                D.append(d)
                offsetK += blockK

        return self.collapse_filter(D, dtype=np.float32), grad_g


@ops.RegisterGradient("BlocksparseConv")
def blocksparse_conv_grad(op, grad):

    overlapC  = op.get_attr("overlapC")
    overlapK  = op.get_attr("overlapK")
    C         = op.get_attr("C")
    K         = op.get_attr("K")
    DHW       = op.get_attr("DHW")
    MPQ       = op.get_attr("MPQ")
    dimF      = op.get_attr("dimF")
    trs       = op.get_attr("trs")
    magic_trs = op.get_attr("magic_trs")
    shift_trs = op.get_attr("shift_trs")
    fshare    = op.get_attr("fshare")
    bshare    = op.get_attr("bshare")
    debug     = op.get_attr("debug")

    assert grad.get_shape()[1] == K

    grad_I = blocksparse_conv_op(
        op.inputs[0], op.inputs[1], op.inputs[2],
        op.inputs[3], op.inputs[4], op.inputs[5],
        op.inputs[6], grad, c_type=grad.dtype,
        mode=1, overlapC=overlapC, overlapK=overlapK,
        C=C, K=K, DHW=DHW, MPQ=MPQ, trs=trs,
        magic_trs=magic_trs, shift_trs=shift_trs,
        dimF=dimF, fshare=fshare, bshare=bshare, debug=debug )

    grad_F = blocksparse_conv_op(
        op.inputs[0], op.inputs[1], op.inputs[2],
        op.inputs[3], op.inputs[4], op.inputs[5],
        grad, op.inputs[7], c_type=grad.dtype,
        mode=2, overlapC=overlapC, overlapK=overlapK,
        C=C, K=K, DHW=DHW, MPQ=MPQ, trs=trs,
        magic_trs=magic_trs, shift_trs=shift_trs,
        dimF=dimF, fshare=fshare, bshare=bshare, debug=debug )

    return (None, None, None, None, None, None, grad_F, grad_I)


@ops.RegisterGradient("L2NormalizeKCTRS")
def blocksparse_l2_normalize_grad_kctrs(op, grad_y, sum_sqr_x):

    epsilon   = op.get_attr("epsilon")
    K         = op.get_attr("K")
    grad_x    = l2_normalize_grad_kctrs(
        grad_y, op.inputs[0], op.outputs[1], op.inputs[1], epsilon=epsilon, K=K)

    return (grad_x, None)

@ops.RegisterGradient("L2NormalizeGainKCTRS")
def blocksparse_l2_normalize_grad_kctrs(op, grad_y, sum_sqr_x):

    epsilon = op.get_attr("epsilon")
    K       = op.get_attr("K")
    grad_x, grad_g = l2_normalize_gain_grad_kctrs(
        grad_y, op.inputs[0], op.inputs[1], op.outputs[1], op.inputs[2], epsilon=epsilon, K=K)

    return (grad_x, grad_g, None)


############################## Blocksparse Deconv #####################################


class BlocksparseDeconv(BlocksparseConv):

    def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", debug=False):

        # C<=>K, DHW<=>MPQ, fprop<=>bprop, update args (EI <=> IE), filter layout: KCTRS <=> CKTRS
        BKC = list()
        for listC, listK in BCK:
            BKC.append([listK, listC])

        if MPQ is None:
            padding = get_padding(padding, TRS, dilates)
            MPQ = [ in_dim(*dims) for dims in zip(TRS, DHW, padding, strides, dilates) ]

        super(BlocksparseDeconv, self).__init__(BKC, TRS, MPQ, DHW, strides, dilates, padding, debug, True)

    def i_shape(self, N): return [N, self.K] + self.MPQ
    def o_shape(self, N): return [N, self.C] + self.DHW

    def fprop_test(self, F, I, alpha=1.0):
        return super(BlocksparseDeconv, self).bprop_test(F, I, alpha)

    def bprop_test(self, F, I, alpha=1.0):
        return super(BlocksparseDeconv, self).fprop_test(F, I, alpha)

    def updat_test(self, E, I, alpha=1.0):
        return super(BlocksparseDeconv, self).updat_test(I, E, alpha, transpose=True)


    def l2_normalize_test(self, F, gain=None, epsilon=1e-12):
        normF = list()
        if gain is None:
            for blockF in F:
                norm = np.sqrt(np.maximum(np.sum(np.square(blockF), axis=(0,2,3,4), keepdims=True), epsilon))
                normF.append((blockF / norm))

        else:
            offsetK = 0
            for blockF in F:
                blockK = blockF.shape[1]
                g = gain[offsetK:offsetK+blockK].reshape((1,blockK,1,1,1))
                norm = np.sqrt(np.maximum(np.sum(np.square(blockF), axis=(0,2,3,4), keepdims=True), epsilon))
                normF.append((g * blockF / norm))
                offsetK += blockK

        return self.collapse_filter(normF, dtype=np.float32)

    def l2_normalize_grad_test(self, F, U, gain=None, epsilon=1e-12):
        D = list()
        if gain is None:
            grad_g = None
            for blockF, blockU in zip(F, U):

                sum_sqr_w = np.sum(np.square(blockF), axis=(0,2,3,4), keepdims=True)
                max_w     = np.maximum(sum_sqr_w, epsilon)
                d         = ( blockU + blockF * (sum_sqr_w >= epsilon) * np.sum(-blockU * blockF / max_w, axis=(0,2,3,4), keepdims=True) ) / np.sqrt(max_w)
                D.append(d)
        else:
            grad_g = np.empty(self.C)
            offsetK = 0
            for blockF, blockU in zip(F, U):
                blockK = blockF.shape[1]
                g = gain[offsetK:offsetK+blockK].reshape((1,blockK,1,1,1))

                sum_sqr_w = np.sum(np.square(blockF), axis=(0,2,3,4), keepdims=True)
                max_w     = np.maximum(sum_sqr_w, epsilon)
                norm_w    = 1.0 / np.sqrt(max_w)

                grad_g[offsetK:offsetK+blockK] = np.sum(blockU * blockF * norm_w, axis=(0,2,3,4))

                d = ( blockU * g + blockF * (sum_sqr_w >= epsilon) * np.sum(-blockU * blockF * g / max_w, axis=(0,2,3,4), keepdims=True) ) * norm_w
                D.append(d)
                offsetK += blockK

        return self.collapse_filter(D, dtype=np.float32), grad_g

    def __call__(self, F, I):
        assert I.get_shape()[1] == self.K
        # mode 0 => 1
        output = blocksparse_deconv_op(
            self.fprop_grid, self.bprop_grid, self.updat_grid,
            self.mpq_lut, self.dhw_lut, self.ck_lut,
            F, I, c_type=I.dtype,
            mode=1, overlapC=self.overlapC, overlapK=self.overlapK,
            C=self.C, K=self.K, DHW=self.DHW, MPQ=self.MPQ, trs=self.trs,
            magic_trs=self.magic_trs[0], shift_trs=self.magic_trs[1],
            dimF=F.get_shape().as_list(), fshare=self.fshared, bshare=self.bshared, debug=self.debug
        )
        return output

    def l2_normalize(self, F, gain=None, epsilon=1e-12, dtype=np.float32):
        if gain is None:
            F, _ = l2_normalize_cktrs(
                F, self.norm_lut, TY=dtype, epsilon=epsilon, K=self.normSize,
                TRS=self.trs, magic_TRS=self.magic_trs[0], shift_TRS=self.magic_trs[1] )
        else:
            assert self.overlapC is False
            F, _ = l2_normalize_gain_cktrs(
                F, gain, self.norm_lut, TY=dtype, epsilon=epsilon, K=self.normSize,
                TRS=self.trs, magic_TRS=self.magic_trs[0], shift_TRS=self.magic_trs[1] )
        return F



@ops.RegisterGradient("BlocksparseDeconv")
def blocksparse_deconv_grad(op, grad):

    overlapC  = op.get_attr("overlapC")
    overlapK  = op.get_attr("overlapK")
    C         = op.get_attr("C")
    K         = op.get_attr("K")
    DHW       = op.get_attr("DHW")
    MPQ       = op.get_attr("MPQ")
    dimF      = op.get_attr("dimF")
    trs       = op.get_attr("trs")
    magic_trs = op.get_attr("magic_trs")
    shift_trs = op.get_attr("shift_trs")
    fshare    = op.get_attr("fshare")
    bshare    = op.get_attr("bshare")
    debug     = op.get_attr("debug")

    # mode 1 => 0
    grad_I = blocksparse_deconv_op(
        op.inputs[0], op.inputs[1], op.inputs[2],
        op.inputs[3], op.inputs[4], op.inputs[5],
        op.inputs[6], grad, c_type=grad.dtype,
        mode=0, overlapC=overlapC, overlapK=overlapK,
        C=C, K=K, DHW=DHW, MPQ=MPQ, trs=trs,
        magic_trs=magic_trs, shift_trs=shift_trs,
        dimF=dimF, fshare=fshare, bshare=bshare, debug=debug )

    # E,I => I,E
    grad_F = blocksparse_deconv_op(
        op.inputs[0], op.inputs[1], op.inputs[2],
        op.inputs[3], op.inputs[4], op.inputs[5],
        op.inputs[7], grad, c_type=grad.dtype,
        mode=2, overlapC=overlapC, overlapK=overlapK,
        C=C, K=K, DHW=DHW, MPQ=MPQ, trs=trs,
        magic_trs=magic_trs, shift_trs=shift_trs,
        dimF=dimF, fshare=fshare, bshare=bshare, debug=debug )

    return (None, None, None, None, None, None, grad_F, grad_I)

@ops.RegisterGradient("L2NormalizeCKTRS")
def blocksparse_l2_normalize_grad_cktrs(op, grad_y, sum_sqr_x):

    epsilon   = op.get_attr("epsilon")
    K         = op.get_attr("K")
    TRS       = op.get_attr("TRS")
    magic_TRS = op.get_attr("magic_TRS")
    shift_TRS = op.get_attr("shift_TRS")

    grad_x  = l2_normalize_grad_cktrs(
        grad_y, op.inputs[0], op.outputs[1], op.inputs[1], epsilon=epsilon,
        K=K, TRS=TRS, magic_TRS=magic_TRS, shift_TRS=shift_TRS)

    return (grad_x, None)

@ops.RegisterGradient("L2NormalizeGainCKTRS")
def blocksparse_l2_normalize_gain_grad_cktrs(op, grad_y, sum_sqr_x):

    epsilon   = op.get_attr("epsilon")
    K         = op.get_attr("K")
    TRS       = op.get_attr("TRS")
    magic_TRS = op.get_attr("magic_TRS")
    shift_TRS = op.get_attr("shift_TRS")

    grad_x, grad_g = l2_normalize_gain_grad_cktrs(
        grad_y, op.inputs[0], op.inputs[1], op.outputs[1], op.inputs[2], epsilon=epsilon,
        K=K, TRS=TRS, magic_TRS=magic_TRS, shift_TRS=shift_TRS)

    return (grad_x, grad_g, None)

############################## ChannelWise Linear #####################################

cwise_linear_op      = _op_module.c_wise_linear
cwise_linear_grad_op = _op_module.c_wise_linear_grad


def cwise_linear(x, gain=None, bias=None, relu=False, bias_first=False, use_tf=False):

    assert gain is not None or bias is not None

    dev = x.op.device.lower()
    if use_tf or not dev or "cpu" in dev:
        if bias_first:
            if bias is not None:
                x += bias
            if gain is not None:
                x *= gain
        else:
            if gain is not None:
                x *= gain
            if bias is not None:
                x += bias
        return tf.nn.relu(x) if relu else x

    gain = [] if gain is None else [gain]
    bias = [] if bias is None else [bias]

    return cwise_linear_op(x, gain, bias, relu=relu, swap=bias_first)


@ops.RegisterGradient("CWiseLinear")
def cwise_linear_axpb_grad(op, dy):

    relu  = op.get_attr("relu")
    swap  = op.get_attr("swap")
    n_a   = op.get_attr("n_a")
    n_b   = op.get_attr("n_b")

    if n_a:
        # anything with a scale factor we need to save the input
        xy = [ op.inputs[0]  ]
    elif relu:
        # with relu(x + b) we save the outputs
        xy = [ op.outputs[0] ]
    else:
        # x + b requires no saved tensors
        xy = []

    a = [ op.inputs[1    ] ] if n_a else []
    b = [ op.inputs[1+n_a] ] if n_b else []

    dx, da, db = cwise_linear_grad_op(dy, xy, a, b, relu=relu, swap=swap)

    if n_a and n_b:
        return dx, da, db
    if n_a:
        return dx, da

    return dx, db

def cwise_linear_test(x, a=1, b=0, relu=False):

    # create broadcastable shapes for a and b
    bcast = list(x.shape)
    for i in range(len(bcast)):
        if i != 1: bcast[i] = 1
    if a is not 1:
        a = a.reshape(bcast)
    if b is not 0:
        b = b.reshape(bcast)

    y = a*x + b
    if relu:
        y = np.maximum(y, 0.)

    return y

def cwise_linear_grad_test(dy, x, a=1, b=0, relu=False):

    bcast = list(dy.shape)
    axis  = list()
    for i in range(len(bcast)):
        if i != 1:
            bcast[i] = 1
            axis.append(i)
    axis = tuple(axis)
    if a is not 1:
        a = a.reshape(bcast)
    if b is not 0:
        b = b.reshape(bcast)

    if relu:
        dy = dy * (a*x + b > 0.0)

    dx = a * dy
    da = np.sum(dy * x, axis=axis)
    db = np.sum(dy,     axis=axis)

    return dx, da, db


############################## Helpers #####################################

def dilation_size(S, dilate):
    return S * dilate - dilate + 1

def tf_out_dim_pad(S, W, padding, stride, dilate):
    S = dilation_size(S, dilate)
    if padding.upper() == "SAME":
        Q = ceil_div(W, stride)
        p = max((Q - 1) * stride + S - W, 0) // 2
    else:
        Q = ceil_div(W - S + 1, stride)
        p = 0;
    return Q, p

def out_dim(S, W, padding, stride, dilate):
    return ceil_div(W - dilation_size(S, dilate) + 1 + 2*padding, stride)

def in_dim(S, W, padding, stride, dilate):
    # Note: inverting ceil_div is ambigous, assume orig numerator was even multiple of stride
    # It's safer to just manually specify the output_dim
    return W*stride + S - 2*padding - (S & 1)

def expand_dims(dim, pad_val=1):
    return [pad_val] * (3 - len(dim)) + list(dim)

def get_padding(padding, TRS, dilates):
    if type(padding) is str:
        if padding.upper() == "SAME":
            padding = [ dilation_size(*dims) // 2 for dims in zip(TRS, dilates) ]
        else:
            padding = [0,0,0]
    else:
        padding = expand_dims(padding, 0)
    return padding

def fprop_lut(q, X, S, padding, stride, dilate):
    qs = q * stride - padding
    image = list()
    for s in range(S):
        x = qs + s * dilate
        image.append(x if x >= 0 and x < X else -1)
    return image

def bprop_lut(x, Q, S, padding, stride, dilate):
    pad_eff = dilation_size(S, dilate) - padding - 1
    xs = x - pad_eff
    image = list()
    # invert the filter to image mapping
    for s in range(S-1, -1, -1):
        q = xs + s * dilate
        if q % stride == 0:
            q //= stride
            if q >= 0 and q < Q:
                image.append(q)
            else:
                image.append(-1)
        else:
            # we need to be able to distinguish a hole in striding and edge padding
            image.append(-2)
    return image

def fprop_slice(q, X, S, padding, stride, dilate):
    qs = q * stride - padding
    x1 = None
    for s in range(S):
        x = qs + s * dilate
        if x1 is None and x >= 0:
            x1 = x
            f1 = s
        if x < X:
            x2 = x
            f2 = s
    return (slice(f1, f2 + 1), slice(x1, x2 + 1, dilate), f2 - f1 + 1)

def bprop_slice(x, Q, S, padding, stride, dilate):
    pad_eff = dilation_size(S, dilate) - padding - 1
    xs = x - pad_eff
    f, e = list(), list()
    for s in range(S):
        q = xs + s * dilate
        if q % stride == 0:
            q //= stride
            if q >= 0 and q < Q:
                f.append(s)
                e.append(q)
    if len(f) == 0:
        return (slice(0, 0, 1), slice(0, 0, 1))
    if len(f) == 1:
        fstride = estride = 1
    else:
        fstride = f[1] - f[0]
        estride = e[1] - e[0]
    return (slice(f[0], f[-1]+1, fstride), slice(e[0], e[-1]+1, estride))