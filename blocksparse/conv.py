
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path
import numpy as np
import tensorflow as tf
from operator import mul, lt
from tensorflow.python.framework import ops
if sys.version_info >= (3, 0):
    from functools import reduce

data_files_path = tf.resource_loader.get_data_files_path()
_op_module = tf.load_op_library(os.path.join(data_files_path, 'blocksparse_ops.so'))

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

cwise_linear_axpb       = _op_module.c_wise_linear_axpb
cwise_linear_ax         = _op_module.c_wise_linear_ax
cwise_linear_xpb        = _op_module.c_wise_linear_xpb
cwise_linear_grad_axpb  = _op_module.c_wise_linear_grad_axpb
cwise_linear_grad_ax    = _op_module.c_wise_linear_grad_ax
cwise_linear_grad_xpb   = _op_module.c_wise_linear_grad_xpb

# float_cast_op           = _op_module.float_cast




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
    def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", edge_bias=False, debug=False, deconv=False):

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

        trs = reduce(mul, TRS, 1)
        dhw = reduce(mul, DHW, 1)
        mpq = reduce(mul, MPQ, 1)

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
            self.mpqLut.append( [ self.fprop_lut( x, DHW[i], *fdata[i]) for x in range(MPQ[i]) ] )
            self.dhwLut.append( [ self.bprop_lut( x, MPQ[i], *fdata[i]) for x in range(DHW[i]) ] )
        mpq_lut = self.spatial_grid(DHW, MPQ, self.mpqLut, mpq, trs)
        dhw_lut = self.spatial_grid(MPQ, DHW, self.dhwLut, dhw, trs)

        # get the super block dimension
        dim_O = mpq_lut.shape[0]
        dim_I = dhw_lut.shape[0]

        # merge the spatial and feature outer product grid info
        fpropGrid = list()
        for dim_K, block_C, block_K, offset_CK, offset_F in fpropGridF:
            for order, idx_MPQ, idx_K in sorted([ (morton(0,o,k), o,k) for o,k in np.ndindex(dim_O, dim_K) ]):
                # idx_K/idx_MPQ, block_K/block_C, offset_CK, offset_F
                fpropGrid.append( [
                    idx_MPQ + (idx_K   << 16),
                    block_C + (block_K << 16),
                    offset_CK, offset_F ] )

        bpropGrid = list()
        for dim_C, block_C, block_K, offset_CK, offset_F in bpropGridF:
            for order, idx_DHW, idx_C in sorted([ (morton(0,i,c), i,c) for i,c in np.ndindex(dim_I, dim_C) ]):
                # idx_C/idx_DHW, block_K/block_C, offset_CK, offset_F
                bpropGrid.append( [
                    idx_DHW + (idx_C   << 16),
                    block_C + (block_K << 16),
                    offset_CK, offset_F ] )

        updatGrid = list()
        for dim_CTRS, dim_K, block_C, block_K, offset_CK, offset_F in updatGridF:
            for order, idx_MPQ, idx_K, idx_CTRS in sorted([ (morton(o,k,c), o,k,c) for o,k,c in np.ndindex(dim_O, dim_K, dim_CTRS) ]):
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

        if edge_bias:
            self.init_edge_bias()
        else:
            self.edgeBiasDim = 0

    def spatial_grid(self, DHW, MPQ, mpqLut, mpq, trs):

        # Find the most efficient super-block using a tile of size 32
        # For ties then pick the larger tile in the W dim (more contiguous memory access)
        # TODO: allow a mixture of superblock shapes, or maybe odd shapes to get better ulilization
        ulilization = list()
                  # xxxxx    yxxxx    yyxxx   zyyxx
        for sb in ((1,1,32),(1,2,16),(1,4,8),(2,4,4)):
            util = float(mpq) / reduce(mul, [ ceil_div(*dims) for dims in zip(MPQ, sb) ], 32)
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
        for order, sb_mpq in sorted([ (morton(*mpq), mpq) for mpq in np.ndindex(*mpqDim) ]):

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

    def _edge_bias_init(self, MPQ, mpqLut):

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

            biasLut = biasHead + biasData
            self.edgeBiasMap = biasMap
            self.edgeBiasLut = tf.constant(np.array(biasLut, dtype=np.int32), name="edge_bias_lut")

    def init_edge_bias(self):
        self._edge_bias_init(self.MPQ, self.mpqLut)

    def edge_bias(self, x, eb):
        if self.edgeBiasDim:
            o_shape = self.o_shape(None)
            return edge_bias_op(x, eb, self.edgeBiasLut, edges=self.edgeBiasDim, K=o_shape[1], MPQ=o_shape[2:])
        else:
            return x

    def edge_bias_shape(self): return (self.K, self.edgeBiasDim)

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


    def __call__(self, F, I, edge_bias=None):
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
        if edge_bias is not None and self.edgeBiasDim:
            output = self.edge_bias(output, edge_bias)
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

    def fprop_lut(self, q, X, S, padding, stride, dilate):
        qs = q * stride - padding
        image = list()
        for s in range(S):
            x = qs + s * dilate
            image.append(x if x >= 0 and x < X else -1)
        return image

    def bprop_lut(self, x, Q, S, padding, stride, dilate):
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

    def fprop_slice(self, q, X, S, padding, stride, dilate):
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

    def bprop_slice(self, x, Q, S, padding, stride, dilate):
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

    def init_slices(self):
        if self.mpqSlice is None:
            self.mpqSlice = list()
            self.dhwSlice = list()
            fdata  = list(zip(self.TRS, self.padding, self.strides, self.dilates))
            for i in range(3):
                self.mpqSlice.append( [ self.fprop_slice(x, self.DHW[i], *fdata[i]) for x in range(self.MPQ[i]) ] )
                self.dhwSlice.append( [ self.bprop_slice(x, self.MPQ[i], *fdata[i]) for x in range(self.DHW[i]) ] )

    def fprop_test(self, F, I, alpha=1.0, edge_bias=None):
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

        O = self.edge_bias_test(O, edge_bias)

        return O

    def bprop_test(self, F, I, alpha=1.0, edge_bias=None):
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

        O = self.edge_bias_test(O, edge_bias)

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

        EBU = self.edge_bias_grad_test(I if transpose else E)

        return self.collapse_filter(U, dtype=np.float32), EBU

    def edge_bias_test(self, x, eb):
        if eb is not None and self.edgeBiasDim:
            N, K = x.shape[0:2]
            y = np.array(x.reshape(N, K, -1))
            for i in range(self.edgeBiasDim):
                y[:,:,self.edgeBiasMap[i]] += eb[:,i].reshape(1, K, 1)
            return y.reshape(x.shape)
        else:
            return x

    def edge_bias_grad_test(self, grad_y):
        if self.edgeBiasDim:
            N, K = grad_y.shape[0:2]
            grad_y = grad_y.reshape(N, K, -1)
            grad_b = np.empty(self.edge_bias_shape())
            for i in range(self.edgeBiasDim):
                grad_b[:,i] = grad_y[:,:,self.edgeBiasMap[i]].sum(axis=(0,2))
            return grad_b
        else:
            return None

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

@ops.RegisterGradient("EdgeBias")
def edge_bias_grad(op, grad):

    edges = op.get_attr("edges")
    K     = op.get_attr("K")
    MPQ   = op.get_attr("MPQ")

    assert grad.get_shape()[1] == K

    grad_b = edge_bias_grad_op(grad, op.inputs[2], edges=edges, K=K, MPQ=MPQ)
    return (grad, grad_b, None)

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

    def __init__(self, BCK, TRS, DHW, MPQ=None, strides=(1,1,1), dilates=(1,1,1), padding="SAME", edge_bias=False, debug=False):

        # C<=>K, DHW<=>MPQ, fprop<=>bprop, update args (EI <=> IE), filter layout: KCTRS <=> CKTRS
        BKC = list()
        for listC, listK in BCK:
            BKC.append([listK, listC])

        if MPQ is None:
            padding = get_padding(padding, TRS, dilates)
            MPQ = [ in_dim(*dims) for dims in zip(TRS, DHW, padding, strides, dilates) ]

        super(BlocksparseDeconv, self).__init__(BKC, TRS, MPQ, DHW, strides, dilates, padding, edge_bias, debug, True)

    def edge_bias_shape(self): return (self.C, self.edgeBiasDim)

    def init_edge_bias(self):
        self._edge_bias_init(self.DHW, self.dhwLut)

    def i_shape(self, N): return [N, self.K] + self.MPQ
    def o_shape(self, N): return [N, self.C] + self.DHW

    def fprop_test(self, F, I, alpha=1.0, edge_bias=None):
        return super(BlocksparseDeconv, self).bprop_test(F, I, alpha, edge_bias)

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

    def __call__(self, F, I, edge_bias=None):
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
        if edge_bias is not None and self.edgeBiasDim:
            output = self.edge_bias(output, edge_bias)
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

def cwise_linear(x, a=None, b=None):

    shape = x.get_shape()
    C     = shape[1]
    DHW   = shape[2:].num_elements()

    if a is not None:
        assert C == a.get_shape().num_elements()
    if b is not None:
        assert C == b.get_shape().num_elements()

    if a is not None and b is not None:
        return cwise_linear_axpb(x, a, b, C=C, DHW=DHW)
    if a is not None:
        return cwise_linear_ax(x, a, C=C, DHW=DHW)
    if b is not None:
        return cwise_linear_xpb(x, b, C=C, DHW=DHW)
    return x

@ops.RegisterGradient("CWiseLinearAXPB")
def cwise_linear_axpb_grad(op, dy):
    C   = op.get_attr("C")
    DHW = op.get_attr("DHW")
    magic = magic64u(DHW)
    return cwise_linear_grad_axpb(dy, op.inputs[0], op.inputs[1], op.inputs[2], C=C, DHW=DHW, magic_DHW=magic[0], shift_DHW=magic[1])

@ops.RegisterGradient("CWiseLinearAX")
def cwise_linear_ax_grad(op, dy):
    C   = op.get_attr("C")
    DHW = op.get_attr("DHW")
    magic = magic64u(DHW)
    return cwise_linear_grad_ax(dy, op.inputs[0], op.inputs[1], C=C, DHW=DHW, magic_DHW=magic[0], shift_DHW=magic[1])

@ops.RegisterGradient("CWiseLinearXPB")
def cwise_linear_xpb_grad(op, dy):
    C   = op.get_attr("C")
    DHW = op.get_attr("DHW")
    magic = magic64u(DHW)
    db = cwise_linear_grad_xpb(dy, op.inputs[1], C=C, DHW=DHW, magic_DHW=magic[0], shift_DHW=magic[1])
    return dy, db

def cwise_linear_test(x, a=1, b=0):

    # create broadcastable shapes for a and b
    shape = list(x.shape)
    for i in range(len(shape)):
        if i != 1: shape[i] = 1
    if a is not 1:
        a = a.reshape(shape)
    if b is not 0:
        b = b.reshape(shape)

    return a*x + b

def cwise_linear_grad_test(dy, x, a=1):

    shape = list(dy.shape)
    axis  = list()
    for i in range(len(shape)):
        if i != 1:
            shape[i] = 1
            axis.append(i)
    axis = tuple(axis)
    if a is not 1:
        a = a.reshape(shape)

    dx = a * dy
    da = np.sum(dy * x, axis=axis)
    db = np.sum(dy,     axis=axis)

    return dx, da, db


############################## Helpers #####################################

def ceil_div(x, y):
    return -(-x // y)

def dilation_size(S, dilate):
    return S * dilate - dilate + 1

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

# Morton ordering (z-order) of 3D coords
def morton(z, y, x):
    answer = 0
    bits = max(len(bin(x)), len(bin(y)), len(bin(z))) - 2
    for i in range(bits):
        mshifted = 1 << i;
        shift = i << 1
        answer |= ((x & mshifted) << shift) | ((y & mshifted) << (shift + 1)) | ((z & mshifted) << (shift + 2))
        #print mshifted, shift, answer, bin(answer)
    return answer

# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 32 bits
# Shamelessly pulled directly from:
# http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
def magic32u(nmax, d):
    nc = ((nmax + 1) // d) * d - 1
    nbits = len(bin(nmax)) - 2
    for p in range(0, 2 * nbits + 1):
        if 2 ** p > nc * (d - 1 - (2 ** p - 1) % d):
            m = (2 ** p + d - 1 - (2 ** p - 1) % d) // d
            return (m, p)
    raise ValueError("Can't find magic number for division")


# Magic numbers and shift amounts for integer division
# Suitable for when nmax*magic fits in 64 bits and the shift
# lops off the lower 32 bits
def magic64u(d):
    # 3 is a special case that only ends up in the high bits
    # if the nmax is 0xffffffff
    # we can't use 0xffffffff for all cases as some return a 33 bit
    # magic number
    nmax = 0xffffffff if d == 3 else 0x7fffffff
    magic, shift = magic32u(nmax, d)
    if magic != 1:
        shift -= 32
    return (magic, shift)