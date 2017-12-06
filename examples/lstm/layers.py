import numpy      as np
import tensorflow as tf
from   sklearn.externals  import joblib
from   blocksparse.matmul import BlocksparseMatMul, SparseProj, group_param_grads, get_parents, add_control_input, largest_block
from   blocksparse.norms  import layer_norm
import blocksparse.ewops  as ew
import masks

from utils import ones_initializer, zeros_initializer, normal_initializer, ortho_initializer, make_path, ceil_div

agg_method=0 # set to 3 when in bfloat16 mode

# Debugging function
def print_act_stats(x, _str="", flatten=False):
    if False:
        return x
    _x = ew.float_cast(x, dtype=tf.float32)
    if flatten:
        _x = tf.reshape(_x, [-1])
    if len(_x.get_shape()) == 1:
        x_mean, x_var = tf.nn.moments(_x, [0], keep_dims=True)
    if len(_x.get_shape()) == 2:
        x_mean, x_var = tf.nn.moments(_x, [0], keep_dims=True)
    if len(_x.get_shape()) == 4:
        x_mean, x_var = tf.nn.moments(_x, [0,2,3], keep_dims=True)
    stats = [tf.reduce_min(x_mean), tf.reduce_mean(x_mean), tf.reduce_max(x_mean),\
            tf.reduce_min(tf.sqrt(x_var)), tf.reduce_mean(tf.sqrt(x_var)), tf.reduce_max(tf.sqrt(x_var))]
    __str = "["+_str+"] "+x.name
    print(__str)
    return tf.Print(x, stats, __str)

class HParams(object):

    no_serialize = set(["feed_dict","params","initializers","restore","state_shape"])

    def __init__(self, args):

        for k, v in args.__dict__.items():
            if type(k) is str and k[0] != '_':
                setattr(self, k, v)

        self.feed_dict = dict()
        self.params    = dict()
        if self.restore:

            state = joblib.load(self.restore)

            for k, v in state.items():
                setattr(self, k, v)

            print("Restore:")
            for name in sorted(list(self.initializers.keys())):
                val = self.initializers[name]
                print(name, val.shape, val.size)
        else:
            self.initializers = dict()

    def get_variable(self, name, shape, initializer):

        scope = tf.get_variable_scope()
        if scope.reuse:
            return tf.get_variable(name)

        ph = tf.placeholder(tf.float32, shape)
        p  = tf.get_variable(name, initializer=ph)

        # add last part of scope to name to allow non-unique names
        name = scope.name.split("/")[-1] + "/" + name

        if name not in self.params:
            self.params[name] = p

        if name not in self.initializers:
            self.initializers[name] = initializer(shape)

        self.feed_dict[ph] = self.initializers[name]

        return p

    def save(self, sess, ema):

        make_path(self.save_path)

        state = dict()

        params = sess.run([ema.average(p) for p in self.params.values()])
        state["initializers"] = dict(zip(self.params.keys(), params))

        for k, v in self.__dict__.items():
            if k not in HParams.no_serialize:
                state[k] = v

        joblib.dump(state, self.save_path)

    def finish_init(self):
        # free up potentially large amount of memory used by these
        self.initializers = None
        self.feed_dict    = None


class LSTM_Model(object):

    def __init__(self, hps, train):

        self.hps    = hps
        self.train  = train
        self.embd   = Embedding(hps, train)
        if hps.lstm_type == 'lstm':
            self.lstm   = LSTM_vanilla(hps, train)
        if hps.lstm_type == 'scottbrain':
            self.lstm   = LSTM_scott(hps, train)
        if hps.lstm_type == 'rnn':
            self.lstm   = RNN(hps, train)

        # do this once for all gpus
        if "bsmm" not in hps.__dict__:
            self.gen_masks()

        self.fc = FullyConnected(hps)


    def forward(self, X, S, Y, ema=None):

        inputs          = self.embd.forward(X, ema=ema)
        outputs, states = self.lstm.forward(inputs, S, ema=ema)
        logits          = self.fc.forward(outputs, ema=ema)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(Y, [-1]))
        loss = tf.reduce_mean(loss)

        # save for layerwise custom gradients
        self.logits = logits
        self.loss = loss

        print("LSTM_Model::forward")

        return loss, states

    def backward(self):

        # Compute gradients 1 layer at a time.
        # This enables memory efficient mode to be implemented.
        fc_grads               = tf.gradients(self.loss, self.logits, aggregation_method=agg_method)
        fc_grads,   lstm_grads = self.fc.backward(fc_grads)
        lstm_grads, embd_grads = self.lstm.backward(lstm_grads)
        embd_grads             = self.embd.backward(embd_grads)

        print("LSTM_Model::backward")

        return fc_grads + lstm_grads + embd_grads

    def gen_masks(self):
        hps = self.hps
        hps.bsmm = bsmm = dict()

        assert hps.nhidden % hps.block_size == 0
        assert hps.nembd % 32 == 0

        # Create block-sparse matmul ops (to be shared by all instances of the model)
        # We only need 1 instance of the lut constants
        with tf.name_scope("BlocksparseMatMul"):

            if hps.nproj_in != hps.nhidden:
                # assume small projection values are acutally strides
                if hps.nproj_in <= hps.block_size * 4:
                    hps.sproj_mul = SparseProj(hps.nhidden, proj_stride=hps.nproj_in)
                    hps.sproj_add = SparseProj(hps.nhidden, proj_stride=hps.nproj_in)
                    hps.nproj_in  = hps.sproj_mul.nproj
                else:
                    hps.sproj_mul = SparseProj(hps.nhidden, nproj=hps.nproj_in)
                    hps.sproj_add = SparseProj(hps.nhidden, nproj=hps.nproj_in)
            else:
                hps.sproj_mul = None
                hps.sproj_add = None

            if hps.nproj_out != hps.nhidden:
                # assume small projection values are acutally strides
                if hps.nproj_out <= hps.block_size * 4:
                    hps.sproj_out = SparseProj(hps.nhidden, proj_stride=hps.nproj_out, block_size=32)
                    hps.nproj_out = hps.sproj_out.nproj
                else:
                    hps.sproj_out = SparseProj(hps.nhidden, nproj=hps.nproj_out)
            else:
                hps.sproj_out = None

            # for the input and output projections, use the largest block size that fits
            blk_in,  nproj_in  = largest_block(hps.nproj_in)
            blk_out, nproj_out = largest_block(hps.nproj_out)

            nhidden = hps.nhidden // hps.block_size
            nembd   = hps.nembd   // blk_in
            nvocab  = ceil_div(hps.nvocab, blk_out)

            # the dense input mask
            mask   = np.ones( (nembd,  nproj_in), dtype=np.int32)
            bsmm["x"] = BlocksparseMatMul(mask, block_size=blk_in, feature_axis=hps.axis, name="lstm_x")

            istep_masks = []
            if hps.share_masks:
                # all gates and internal steps get the same mask
                mask = masks.make_mask(n=nhidden, kind=hps.sparsity)
                bsmm_p = BlocksparseMatMul(mask, block_size=hps.block_size, feature_axis=hps.axis, name="lstm_h")

                for p in list("ifou") + ["h%d" % i for i in range(hps.isteps)]:
                    bsmm[p] = bsmm_p

                istep_masks = [ mask for i in  range(hps.isteps + 1)]
            else:
                # internal steps get different masks
                for p in ["h%d" % i for i in range(hps.isteps)]:
                    mask = masks.make_mask(n=nhidden, kind=hps.sparsity)
                    bsmm[p] = BlocksparseMatMul(mask, block_size=hps.block_size, feature_axis=hps.axis, name="lstm_%s" % p)
                    istep_masks.append(mask)

                # gates get the same mask (TODO: experiment here with differen masks)
                mask = masks.make_mask(n=nhidden, kind=hps.sparsity)
                bsmm_p = BlocksparseMatMul(mask, block_size=hps.block_size, feature_axis=hps.axis, name="lstm_g")
                for p in list("ifou"):
                    bsmm[p] = bsmm_p

                istep_masks.append(mask)

            # the output mask
            mask   = np.ones( (nproj_out, nvocab), dtype=np.int32)
            bsmm["y"] = BlocksparseMatMul(mask, block_size=blk_out, feature_axis=hps.axis, name="lstm_o")

            hps.mix_factor = masks.mix_factor(istep_masks)
            hps.sparsity  += " (%.4f%%)" % (100.0 * bsmm["u"].sparsity)


class Embedding(object):
    def __init__(self, hps, train, scope='embedding'):
        self.hps    = hps
        self.train  = train
        self.scope  = scope

    def forward(self, x, ema=None):
        hps = self.hps

        assert hps.nsteps % hps.x_group_size == 0
        xgroups = hps.nsteps // hps.x_group_size

        with tf.variable_scope(self.scope):

            w = hps.get_variable("w", [hps.nvocab, hps.nembd], ortho_initializer())
            g = hps.get_variable("g", [hps.nvocab,         1],  ones_initializer())

            self.params = [w, g]

            if ema is not None:
                w = ema.average(w)
                g = ema.average(g)

            w = tf.nn.l2_normalize(w, dim=1) * g

            # x (nsteps, nbatch)
            # w (nvocab, nembd)
            # o (nsteps, nbatch, nembd)
            words = tf.nn.embedding_lookup(w, x)
            if self.train and hps.dropout > 0 and hps.dropout_input > 0:
                words = tf.nn.dropout(words, 1.-hps.dropout, [hps.nsteps, hps.batch_size, 1])

            # potentially down cast to fp16 to save memory and speed things up
            #words = ew.float_cast(words, dtype=hps.dtype)

            # (x_group_size x nbatch, nembd) * xgroups
            outputs = [tf.reshape(x, [-1, hps.nembd]) for x in tf.split(words, xgroups, 0)]
            if hps.axis == 0:
                outputs = [tf.transpose(x) for x in outputs]

            self.outputs = [ew.float_cast(x, dtype=hps.dtype) for x in outputs]

            outputs = [tf.stop_gradient(x) for x in self.outputs]

        return outputs

    def backward(self, grad_ys):
        param_grads = tf.gradients(self.outputs, self.params, grad_ys, aggregation_method=agg_method)

        return list(zip(param_grads, self.params))


class FullyConnected(object):
    def __init__(self, hps, scope='fc'):
        self.hps    = hps
        self.scope  = scope

    def forward(self, inputs, ema=None):
        hps     = self.hps
        bsmm    = hps.bsmm
        xgroup  = hps.x_group_size
        xgroups = len(inputs) // xgroup
        sproj   = hps.sproj_out

        self.inputs = inputs

        if sproj is not None:
            inputs = [ sproj.gather(h) for h in inputs ]

        with tf.variable_scope(self.scope):

            w = hps.get_variable("w", bsmm["y"].w_shape, normal_initializer())
            g = hps.get_variable("g", [hps.nvocab],  ones_initializer())
            b = hps.get_variable("b", [hps.nvocab], zeros_initializer())

            self.params = [w, g, b]

            if ema is not None:
                w = ema.average(w)
                g = ema.average(g)
                b = ema.average(b)

            #w = ew.float_cast(w, dtype=hps.dtype)
            w = bsmm["y"].l2_normalize(w, dtype=hps.dtype)

            # compute the fc matmul in groups for better memory efficiency.
            ygroups = []
            for i in range(xgroups):
                x = tf.concat(inputs[i*xgroup:(i+1)*xgroup], 1 - hps.axis)

                # (nsteps x nbatch, nvocab) = (nsteps x nbatch, hidden) . (nhidden, nvocab)
                ygroups.append(bsmm["y"](x, w, dw_dtype=hps.dw_dtype))

            y = tf.concat(ygroups, 1 - hps.axis)

            # cast to float32 before entering cost function
            y = ew.float_cast(y, dtype=tf.float32, dx_dtype=hps.dx_dtype)

            if hps.axis == 0:
                y = tf.transpose(y)

            if (hps.nvocab % 32) != 0:
                y = tf.slice(y, [0,0], [-1, hps.nvocab])

            self.outputs = y*g + b

            outputs = tf.stop_gradient(self.outputs)

        return outputs

    def backward(self, grad_ys):

        nparams = len(self.params)

        grads = tf.gradients(self.outputs, self.params + self.inputs, grad_ys, aggregation_method=agg_method)

        param_grads = grads[0:nparams]
        input_grads = grads[nparams:]

        grads = list(zip(param_grads, self.params))

        return grads, input_grads

class LSTM_vanilla(object):
    # this model is currently broken with way masks are currently initialized
    def __init__(self, hps, train, scope='lstm'):

        self.hps    = hps
        self.train  = train
        self.scope  = scope

    def forward(self, inputs, states, ema=None):

        hps  = self.hps
        bsmm = hps.bsmm

        with tf.variable_scope(self.scope) as scope:

            self.param_names = ['xi','xf','xo','xu','hi','hf','ho','hu']
            self.params = dict()

            for p in self.param_names:

                if 'x' in p:
                    bsmm_p, size = (bsmm.x, hps.nproj_in)

                elif 'h' in p:
                    bsmm_p, size = (bsmm.h, hps.nhidden)

                b_init = ones_initializer(hps.forget_bias) if p == 'hf' else zeros_initializer()

                w = hps.get_variable("w_" + p, bsmm_p.w_shape, bsmm_p.identity_init())
                g = hps.get_variable("g_" + p, [size], ones_initializer())
                b = hps.get_variable("b_" + p, [size], b_init)

                if ema is not None:
                    w = ema.average(w)
                    g = ema.average(g)
                    b = ema.average(b)

                wc = ew.float_cast(w, dtype=hps.dtype)

                self.params[p] = (wc, g, b, w)

            c, h = tf.unstack(states, num=2)
            c  = ew.float_cast(c, dtype=hps.dtype)
            h  = ew.float_cast(h, dtype=hps.dtype)

            xi_w, xi_g, xi_b = self.params["xi"][0:3]
            xf_w, xf_g, xf_b = self.params["xf"][0:3]
            xo_w, xo_g, xo_b = self.params["xo"][0:3]
            xu_w, xu_g, xu_b = self.params["xu"][0:3]

            self.inputs    = inputs
            self.outputs   = []
            self.segments  = []

            for xgroup in inputs:

                if hps.recompute and self.train:
                    # We compute gradient one segment at a time, so prevent tf.gradients from going too far.
                    # We also want to add control inputs to the start of the segment so having wrappers
                    # around the segment inputs is handy.
                    seg = [(tf.stop_gradient(c),tf.stop_gradient(h))]
                    self.segments.append(seg)

                # delay input expansion to just prior to use (saves memory)
                with tf.control_dependencies([h]):
                    xwi = bsmm.x(xgroup, xi_w, dw_dtype=hps.dw_dtype)
                    xwf = bsmm.x(xgroup, xf_w, dw_dtype=hps.dw_dtype)
                    xwo = bsmm.x(xgroup, xo_w, dw_dtype=hps.dw_dtype)
                    xwu = bsmm.x(xgroup, xu_w, dw_dtype=hps.dw_dtype)

                xwi = tf.split(xwi, hps.x_group_size, 1 - hps.axis)
                xwf = tf.split(xwf, hps.x_group_size, 1 - hps.axis)
                xwo = tf.split(xwo, hps.x_group_size, 1 - hps.axis)
                xwu = tf.split(xwu, hps.x_group_size, 1 - hps.axis)

                masks = []
                for xi, xf, xo, xu in zip(xwi, xwf, xwo, xwu):
                    xi = layer_norm(xi, xi_g, xi_b, axis=hps.axis)
                    xf = layer_norm(xf, xf_g, xf_b, axis=hps.axis)
                    xo = layer_norm(xo, xo_g, xo_b, axis=hps.axis)
                    xu = layer_norm(xu, xu_g, xu_b, axis=hps.axis)

                    c, h, mask = self.cell(c, h, xi, xf, xo, xu)
                    _masks = [mask]
                    for _ in range(1, hps.lsteps):
                        c, h, mask = self.cell(c, h, None, None, None, None)
                        _masks.append(mask)
                    masks.append(_masks)

                    self.outputs.append(h)

                if hps.recompute and self.train:
                    with tf.name_scope("f_seg_%04d_%d" % (len(self.segments)-1, len(seg)-1)):

                        c_seg, h_seg = seg[0]

                        with tf.control_dependencies([ h_seg ]):
                            xwi = bsmm.x(xgroup, xi_w, dw_dtype=hps.dw_dtype)
                            xwf = bsmm.x(xgroup, xf_w, dw_dtype=hps.dw_dtype)
                            xwo = bsmm.x(xgroup, xo_w, dw_dtype=hps.dw_dtype)
                            xwu = bsmm.x(xgroup, xu_w, dw_dtype=hps.dw_dtype)

                        xwi = tf.split(xwi, hps.x_group_size, 1 - hps.axis)
                        xwf = tf.split(xwf, hps.x_group_size, 1 - hps.axis)
                        xwo = tf.split(xwo, hps.x_group_size, 1 - hps.axis)
                        xwu = tf.split(xwu, hps.x_group_size, 1 - hps.axis)

                        for xi, xf, xo, xu, mask in zip(xwi, xwf, xwo, xwu, masks):
                            xi = layer_norm(xi, xi_g, xi_b, axis=hps.axis)
                            xf = layer_norm(xf, xf_g, xf_b, axis=hps.axis)
                            xo = layer_norm(xo, xo_g, xo_b, axis=hps.axis)
                            xu = layer_norm(xu, xu_g, xu_b, axis=hps.axis)

                            c_seg, h_seg, _ = self.cell(c_seg, h_seg, xi, xf, xo, xu, mask[0])
                            for i in range(1, hps.lsteps):
                                c_seg, h_seg, _ = self.cell(c_seg, h_seg, None, None, None, None, mask[i])

                            seg.append((c_seg, h_seg))

            c = ew.float_cast(c, dtype=tf.float32)
            h = ew.float_cast(h, dtype=tf.float32)
            states = tf.stack([c, h], 0)

            # We calculate the gradient internally.
            # Don't let other layer's gradients flow into here.
            # This is possible because the last cell has free c and h
            # params that are popluated with zeros in the gradients pass.
            outputs = [tf.stop_gradient(x) for x in self.outputs]

        return outputs, states

    def linear(self, p, h, relu=False):
        hps = self.hps
        w, g, b = self.params[p][0:3]
        h = hps.bsmm.h(h, w, dw_dtype=hps.dw_dtype)
        return layer_norm(h, g, b, relu=relu, axis=hps.axis)

    def cell(self, c, h, xi, xf, xo, xu, mask=None):
        hps = self.hps

        assert hps.isteps >= 2, "multiply and add steps of mLSTM require 2 internal steps"

        '''
        for step in range(hps.isteps):

            # we can share one set of params for all isteps
            p = "h%d" % (0 if hps.share_isteps else step)

            if step == 0:
                h = self.linear(p, h)
                if hps.sproj_add is None:
                    h = ew.multiply(h, m)
                else:
                    h = hps.sproj_add.scatter_mul(h, m)
            elif step == 1:
                h = self.linear(p, h)
                if hps.sproj_mul is None:
                    h = ew.add(h, a)
                else:
                    h = hps.sproj_mul.scatter_add(h, a)
                h = ew.relu(h)
            else:
                h = self.linear(p, h, relu=True)
        '''

        i = self.linear("hi", h)
        f = self.linear("hf", h)
        o = self.linear("ho", h)
        u = self.linear("hu", h)

        # apply update dropout, saving mask if we need to recompute forward pass
        if self.train and hps.dropout > 0:
            if mask is None:
                u, mask = ew.dropout(u, keep_prob=1.0-hps.dropout)
            else:
                u = ew.dropout(u, mask=mask)
        else:
            mask = None

        if xi is not None:
            i = ew.add(i, xi)
            f = ew.add(f, xf)
            o = ew.add(o, xo)
            u = ew.add(u, xu)

        c, h = ew.fused_lstm_gates(c, i, f, o, u)
        return c, h, mask

        # i = ew.sigmoid(i)
        # f = ew.sigmoid(f)
        # o = ew.sigmoid(o)
        # u = ew.tanh(u)
        # c = ew.add(ew.multiply(f, c), ew.multiply(i, u))
        # h = ew.multiply(o, ew.tanh(c))
        # return (c, h)

    def backward(self, grad_ys):
        hps = self.hps
        w_params = []
        g_params = []
        b_params = []
        for p in self.param_names:
            g, b, w = self.params[p][1:4]
            w_params.append(w)
            g_params.append(g)
            b_params.append(b)
        params    = w_params + g_params + b_params
        nparams   = len(params)
        nsegments = len(self.segments)

        # memory efficient gradients by recomputing forward pass
        if nsegments > 0:
            param_steps = []
            input_grads = []
            for i in range(nsegments-1,-1,-1):
                with tf.name_scope("b_seg_%04d" % i):

                    h_grads = grad_ys[i*hps.recompute : (i+1)*hps.recompute]
                    if i == nsegments-1:
                        c_grad = tf.zeros(h_grads[0].get_shape())
                    else:
                        fc_matmul_op = get_parents(h_grads[0], "BlocksparseMatmulDX")[0]

                        # delay matmul to avoid memory expansion till just prior to use
                        add_control_input(fc_matmul_op, h_grad.op)

                        h_grads[-1] = ew.add(h_grads[-1], h_grad)

                    s = self.segments[i]
                    x = self.inputs.pop()
                    c_prev, h_prev = s[0]
                    c_next = s[-1][0]
                    h_next = [ seg[1] for seg in s[1:] ]

                    # ensure the forward segments are computed in the backward pass only.
                    add_control_input(c_prev.op, h_grads[-1].op)
                    add_control_input(h_prev.op, h_grads[-1].op)

                    grads = tf.gradients( [c_next] + h_next, params + [c_prev, h_prev, x], [c_grad] + h_grads, aggregation_method=agg_method)

                    param_steps.append(grads[0:nparams])
                    c_grad = grads[nparams+0]
                    h_grad = grads[nparams+1]
                    input_grads.insert(0, grads[nparams+2])

                    #h_grad = tf.check_numerics(h_grad, "h_grad "+str(i)+"/"+str(nsegments))
                    #c_grad = tf.check_numerics(c_grad, "c_grad "+str(i)+"/"+str(nsegments))
                    #input_grads[0] = tf.check_numerics(input_grads[0], "input_grad "+str(i))

            param_grads = []
            for i in range(nparams):
                param_grads.append(tf.add_n([ g[i] for g in param_steps]))

        # Normal gradients for small models
        else:
            grads = tf.gradients(self.outputs, params + self.inputs, grad_ys, aggregation_method=agg_method)
            param_grads = grads[0:nparams]
            input_grads = grads[nparams:]

        # group param grad matmuls to efficinetly accumulate
        if False:
            for i, p in enumerate(self.param_names):
                # a and m are already grouped
                if 'x' not in p:
                    param_grads[i] = group_param_grads(param_grads[i])


        # debug
        if False:
            for i, p in enumerate(self.param_names):
                n = len(self.param_names)
                param_grads[i+0*n] = tf.check_numerics(param_grads[i+0*n], p+" w")
                param_grads[i+1*n] = tf.check_numerics(param_grads[i+1*n], p+" g")
                param_grads[i+1*n] = tf.check_numerics(param_grads[i+2*n], p+" b")

            for i, p in enumerate(input_grads):
                input_grads[i] = tf.check_numerics(input_grads[i], "input_grads "+str(i))


        grads = list(zip(param_grads, params))

        return grads, input_grads

class LSTM_scott(object):

    def __init__(self, hps, train, scope='lstm'):

        self.hps    = hps
        self.train  = train
        self.scope  = scope

    def forward(self, inputs, states, ema=None):

        hps  = self.hps
        bsmm = hps.bsmm

        with tf.variable_scope(self.scope) as scope:

            self.param_names = list("amifou")
            for i in range(1 if hps.share_isteps else hps.isteps):
                self.param_names.append("h%d" % i)

            self.params = dict()

            for p in self.param_names:

                bsmm_p, size = (bsmm["x"], hps.nproj_in) if p in "am" else (bsmm[p], hps.nhidden)

                b_init = ones_initializer() if p == 'f'  else zeros_initializer()

                w = hps.get_variable("w_" + p, bsmm_p.w_shape, bsmm_p.identity_init())
                g = hps.get_variable("g_" + p, [size], ones_initializer())
                b = hps.get_variable("b_" + p, [size], b_init)

                if ema is not None:
                    w = ema.average(w)
                    g = ema.average(g)
                    b = ema.average(b)

                wc = ew.float_cast(w, dtype=hps.dtype)

                self.params[p] = (wc, g, b, w)

            c, h = tf.unstack(states, num=2)
            c  = ew.float_cast(c, dtype=hps.dtype)
            h  = ew.float_cast(h, dtype=hps.dtype)

            wm, gm, bm = self.params["m"][0:3]
            wa, ga, ba = self.params["a"][0:3]

            self.inputs    = inputs
            self.outputs   = []
            self.segments  = []
            for xgroup in inputs:

                if hps.recompute and self.train:
                    # We compute gradient one segment at a time, so prevent tf.gradients from going too far.
                    # We also want to add control inputs to the start of the segment so having wrappers
                    # around the segment inputs is handy.
                    seg = [(tf.stop_gradient(c),tf.stop_gradient(h))]
                    self.segments.append(seg)

                # delay input expansion to just prior to use (saves memory)
                with tf.control_dependencies([h]):
                    xwm = bsmm["x"](xgroup, wm, dw_dtype=hps.dw_dtype)
                    xwa = bsmm["x"](xgroup, wa, dw_dtype=hps.dw_dtype)

                xwm = tf.split(xwm, hps.x_group_size, 1 - hps.axis)
                xwa = tf.split(xwa, hps.x_group_size, 1 - hps.axis)

                masks = []
                for m, a in zip(xwm, xwa):
                    m = layer_norm(m, gm, bm, axis=hps.axis)
                    a = layer_norm(a, ga, ba, axis=hps.axis)

                    c, h, mask = self.cell(c, h, m, a)
                    _masks = [mask]
                    for _ in range(1, hps.lsteps):
                        c, h, mask = self.cell(c, h, None, None)
                        _masks.append(mask)
                    masks.append(_masks)

                    self.outputs.append(h)

                if hps.recompute and self.train:
                    with tf.name_scope("f_seg_%04d_%d" % (len(self.segments)-1, len(seg)-1)):

                        c_seg, h_seg = seg[0]

                        with tf.control_dependencies([ h_seg ]):
                            xwm = bsmm["x"](xgroup, wm, dw_dtype=hps.dw_dtype)
                            xwa = bsmm["x"](xgroup, wa, dw_dtype=hps.dw_dtype)
                        xwm = tf.split(xwm, hps.x_group_size, 1 - hps.axis)
                        xwa = tf.split(xwa, hps.x_group_size, 1 - hps.axis)

                        for m, a, mask in zip(xwm, xwa, masks):
                            m = layer_norm(m, gm, bm, axis=hps.axis)
                            a = layer_norm(a, ga, ba, axis=hps.axis)

                            c_seg, h_seg, _ = self.cell(c_seg, h_seg, m, a, mask[0])
                            for i in range(1, hps.lsteps):
                                c_seg, h_seg, _ = self.cell(c_seg, h_seg, None, None, mask[i])

                            seg.append((c_seg, h_seg))


            c = ew.float_cast(c, dtype=tf.float32)
            h = ew.float_cast(h, dtype=tf.float32)
            states = tf.stack([c, h], 0)

            # We calculate the gradient internally.
            # Don't let other layer's gradients flow into here.
            # This is possible because the last cell has free c and h
            # params that are popluated with zeros in the gradients pass.
            outputs = [tf.stop_gradient(x) for x in self.outputs]

        return outputs, states

    def linear(self, p, h, relu=False):
        hps = self.hps
        w, g, b = self.params[p][0:3]
        h = hps.bsmm[p](h, w, dw_dtype=hps.dw_dtype)
        return layer_norm(h, g, b, relu=relu, axis=hps.axis)

    def cell(self, c, h, m, a, mask=None):
        hps = self.hps

        assert hps.isteps >= 2, "multiply and add steps of mLSTM require 2 internal steps"

        for step in range(hps.isteps):

            # we can share one set of params for all isteps
            p = "h%d" % (0 if hps.share_isteps else step)

            if step == 0:
                h = self.linear(p, h)
                if m is not None:
                    if hps.sproj_add is None:
                        h = ew.multiply(h, m)
                    else:
                        h = hps.sproj_add.scatter_mul(h, m)
            elif step == 1:
                h = self.linear(p, h)
                if a is not None:
                    if hps.sproj_mul is None:
                        h = ew.add(h, a)
                    else:
                        h = hps.sproj_mul.scatter_add(h, a)
                h = ew.relu(h)
            else:
                h = self.linear(p, h, relu=True)

        i = self.linear("i", h)
        f = self.linear("f", h)
        o = self.linear("o", h)
        u = self.linear("u", h)

        # apply update dropout, saving mask if we need to recompute forward pass
        if self.train and hps.dropout > 0:
            if mask is None:
                u, mask = ew.dropout(u, keep_prob=1.0-hps.dropout)
            else:
                u = ew.dropout(u, mask=mask)
        else:
            mask = None

        c, h = ew.fused_lstm_gates(c, i, f, o, u)
        return c, h, mask

        # i = ew.sigmoid(i)
        # f = ew.sigmoid(f)
        # o = ew.sigmoid(o)
        # u = ew.tanh(u)
        # c = ew.add(ew.multiply(f, c), ew.multiply(i, u))
        # h = ew.multiply(o, ew.tanh(c))
        # return (c, h)

    def backward(self, grad_ys):
        hps = self.hps
        w_params = []
        g_params = []
        b_params = []
        for p in self.param_names:
            g, b, w = self.params[p][1:4]
            w_params.append(w)
            g_params.append(g)
            b_params.append(b)
        params    = w_params + g_params + b_params
        nparams   = len(params)
        nsegments = len(self.segments)

        # memory efficient gradients by recomputing forward pass
        if nsegments > 0:
            param_steps = []
            input_grads = []
            for i in range(nsegments-1,-1,-1):
                with tf.name_scope("b_seg_%04d" % i):

                    h_grads = grad_ys[i*hps.recompute : (i+1)*hps.recompute]
                    if i == nsegments-1:
                        c_grad = tf.zeros(h_grads[0].get_shape())
                    else:
                        fc_matmul_op = get_parents(h_grads[0], "BlocksparseMatmulDX")[0]

                        # delay matmul to avoid memory expansion till just prior to use
                        add_control_input(fc_matmul_op, h_grad.op)

                        h_grads[-1] = ew.add(h_grads[-1], h_grad)

                    s = self.segments[i]
                    x = self.inputs.pop()
                    c_prev, h_prev = s[0]
                    c_next = s[-1][0]
                    h_next = [ seg[1] for seg in s[1:] ]

                    # ensure the forward segments are computed in the backward pass only.
                    add_control_input(c_prev.op, h_grads[-1].op)
                    add_control_input(h_prev.op, h_grads[-1].op)

                    grads = tf.gradients( [c_next] + h_next, params + [c_prev, h_prev, x], [c_grad] + h_grads, aggregation_method=agg_method)

                    param_steps.append(grads[0:nparams])
                    c_grad = grads[nparams+0]
                    h_grad = grads[nparams+1]
                    input_grads.insert(0, grads[nparams+2])

            param_grads = []
            for i in range(nparams):
                param_grads.append(tf.add_n([ g[i] for g in param_steps]))

        # Normal gradients for small models
        else:
            grads = tf.gradients(self.outputs, params + self.inputs, grad_ys, aggregation_method=agg_method)
            param_grads = grads[0:nparams]
            input_grads = grads[nparams:]

        # group param grad matmuls to efficinetly accumulate
        for i, p in enumerate(self.param_names):
            # a and m are already grouped
            if p not in 'am':
                param_grads[i] = group_param_grads(param_grads[i])

        grads = list(zip(param_grads, params))

        return grads, input_grads


class RNN(object):

    def __init__(self, hps, train, scope='rnn'):

        self.hps    = hps
        self.train  = train
        self.scope  = scope

    def forward(self, inputs, states, ema=None):

        hps  = self.hps
        bsmm = hps.bsmm

        with tf.variable_scope(self.scope) as scope:

            self.param_names = list("am")
            for i in range(1 if hps.share_isteps else hps.isteps):
                self.param_names.append("h%d" % i)

            self.params = dict()

            for p in self.param_names:

                bsmm_p, size = (bsmm["x"], hps.nproj_in) if p in "am" else (bsmm[p], hps.nhidden)

                w = hps.get_variable("w_" + p, bsmm_p.w_shape, bsmm_p.identity_init())
                g = hps.get_variable("g_" + p, [size], ones_initializer())
                b = hps.get_variable("b_" + p, [size], zeros_initializer())

                if ema is not None:
                    w = ema.average(w)
                    g = ema.average(g)
                    b = ema.average(b)

                wc = ew.float_cast(w, dtype=hps.dtype)

                self.params[p] = (wc, g, b, w)

            c, h = tf.unstack(states, num=2)
            h  = ew.float_cast(h, dtype=hps.dtype)

            wm, gm, bm = self.params["m"][0:3]
            wa, ga, ba = self.params["a"][0:3]

            self.inputs    = inputs
            self.outputs   = []
            self.segments  = []
            for xgroup in inputs:

                # delay input expansion to just prior to use (saves memory)
                with tf.control_dependencies([h]):
                    xwm = bsmm["x"](xgroup, wm, dw_dtype=hps.dw_dtype)
                    xwa = bsmm["x"](xgroup, wa, dw_dtype=hps.dw_dtype)

                xwm = tf.split(xwm, hps.x_group_size, 1 - hps.axis)
                xwa = tf.split(xwa, hps.x_group_size, 1 - hps.axis)

                masks = []
                for m, a in zip(xwm, xwa):
                    m = layer_norm(m, gm, bm, axis=hps.axis)
                    a = layer_norm(a, ga, ba, axis=hps.axis)
                    h = self.cell(h, m, a)

                    self.outputs.append(h)

            h = ew.float_cast(h, dtype=tf.float32)
            states = tf.stack([c, h], 0)

            # We calculate the gradient internally.
            # Don't let other layer's gradients flow into here.
            # This is possible because the last cell has free c and h
            # params that are popluated with zeros in the gradients pass.
            outputs = [tf.stop_gradient(x) for x in self.outputs]

        return outputs, states

    def linear(self, p, h, relu=False):
        hps = self.hps
        w, g, b = self.params[p][0:3]
        h = hps.bsmm[p](h, w, dw_dtype=hps.dw_dtype)
        return layer_norm(h, g, b, relu=relu, axis=hps.axis)

    def cell(self, h, m, a):
        hps = self.hps
        assert hps.isteps >= 2, "multiply and add steps of mLSTM require 2 internal steps"

        for step in range(hps.isteps):

            # we can share one set of params for all isteps
            p = "h%d" % (0 if hps.share_isteps else step)

            if step == 0:
                h = self.linear(p, h)
                if hps.sproj_add is None:
                    h = ew.multiply(h, m)
                else:
                    h = hps.sproj_add.scatter_mul(h, m)
            elif step == 1:
                h = self.linear(p, h)
                if hps.sproj_mul is None:
                    h = ew.add(h, a)
                else:
                    h = hps.sproj_mul.scatter_add(h, a)
                h = ew.relu(h)
            else:
                h = self.linear(p, h, relu=True)

        return h

    def backward(self, grad_ys):
        hps = self.hps
        w_params = []
        g_params = []
        b_params = []
        for p in self.param_names:
            g, b, w = self.params[p][1:4]
            w_params.append(w)
            g_params.append(g)
            b_params.append(b)
        params    = w_params + g_params + b_params
        nparams   = len(params)

        # Normal gradients for small models
        grads = tf.gradients(self.outputs, params + self.inputs, grad_ys, aggregation_method=agg_method)
        param_grads = grads[0:nparams]
        input_grads = grads[nparams:]

        # group param grad matmuls to efficinetly accumulate
        for i, p in enumerate(self.param_names):
            # a and m are already grouped
            if p not in 'am':
                param_grads[i] = group_param_grads(param_grads[i])

        grads = list(zip(param_grads, params))

        return grads, input_grads


def nodesort(ops):
  return sorted(ops, key=lambda op: op.name)


def print_graph(tsort=True):

    g = tf.get_default_graph()

    if tsort:
        from toposort import toposort

        control_outputs = dict()
        for op in g.get_operations():
            for control_input in op._control_inputs:
                if control_input in control_outputs:
                    control_outputs[control_input].append(op)
                else:
                    control_outputs[control_input] = [op]

        def children(op):
          result = set(op for out in op.outputs for op in out.consumers())
          if op in control_outputs:
            result.update(control_outputs[op])
          return result

        deps = dict()
        for op in g.get_operations():
            deps[op] = children(op)
        graph = toposort(deps)

        for ops in graph:
            for op in nodesort(ops):
                print(op.name)
                if "Add" in op.name:
                    for i in op.inputs:
                        print("    " + i.name)
                    print("")

    else:
        for op in g.get_operations():
            print(op.name)
            if "Add" in op.name:
                for i in op.inputs:
                    print("    " + i.name)
                print("")


'''
Adam optimizer
'''
def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999, epsilon=1e-8, gamma=0.):

    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params, aggregation_method=agg_method)
    else:
        grads = cost_or_grads

    t = tf.Variable(1., 'adam_t')
    lr_t = lr * tf.sqrt((1. - tf.pow(mom2, t))) /  (1. - tf.pow(mom1, t))
    updates.append(t.assign_add(1))

    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1 * v + (1. - mom1) * g
            updates.append(v.assign(v_t))
        else:
            v_t = g
        mg_t = mom2 * mg + (1. - mom2) * tf.square(g)
        delta_t = v_t / (tf.sqrt(mg_t) + epsilon)
        if gamma > 0:
            if gamma == 1:
                delta_t *= tf.maximum(1., abs(p))
            else:
                delta_t *= tf.maximum(gamma, abs(p))/gamma
        p_t = p - lr_t * delta_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    return tf.group(*updates)


'''
Adamax optimizer
'''
def adamax_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params, aggregation_method=agg_method)
    else:
        grads = cost_or_grads

    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adamax_mg')
        if mom1 > 0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adamax_v')
            v_t = mom1 * v + (1. - mom1) * g
            updates.append(v.assign(v_t))
        else:
            v_t = g
        mg_t = tf.maximum(mom2 * mg, abs(g))
        p_t = p - lr * v_t / (mg_t + 1e-8)
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    return tf.group(*updates)
