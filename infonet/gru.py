""" An adaptation of chainer's GRU that can handle variable length sequences2arrays.

Pretty much a complete rewrite.

http://docs.chainer.org/en/stable/_modules/chainer/links/connection/gru.html#StatefulGRU
"""
import numpy as np
import chainer as ch

class GRU(ch.Link):
    """ A stateful GRU implementation.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.
    """
    def __init__(self, n_units, n_inputs=None,
                 W_init=None, U_init=None, bias_init=None):
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(GRU, self).__init__(
            W_r=(n_inputs, n_units),
            U_r=(n_units, n_units),
            b_r=(n_units,),
            W_z=(n_inputs, n_units),
            U_z=(n_units, n_units),
            b_z=(n_units,),
            W_h=(n_inputs, n_units),
            U_h=(n_units, n_units),
            b_h=(n_units,)
        )
        # initialize params
        if not W_init:
            W_init = ch.initializers.GlorotUniform(scale=1.0)
        W_init(self.W_r.data)
        W_init(self.W_z.data)
        W_init(self.W_h.data)
        if not U_init:
            U_init = ch.initializers.Orthogonal(scale=1.0)
        U_init(self.U_r.data)
        U_init(self.U_z.data)
        U_init(self.U_h.data)
        if not bias_init:
            bias_init = ch.initializers.Zero()
        bias_init(self.b_r.data)
        bias_init(self.b_z.data)
        bias_init(self.b_h.data)

        # concat W's and U's for matmul speed
        self.WU_r = ch.functions.vstack([self.W_r, self.U_r])
        self.WU_z = ch.functions.vstack([self.W_z, self.U_z])
        self.WU_h = ch.functions.vstack([self.W_h, self.U_h])

        self.reset_state()

    def set_state(self, array):
        self.h = ch.Variable(array)

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        # function shorthand
        sigmoid = ch.functions.sigmoid
        tanh = ch.functions.tanh
        matmul = ch.functions.matmul
        hstack = ch.functions.hstack
        bcast_to = ch.functions.broadcast_to

        # set the state if needed
        batch_size = x.shape[0]
        state_shape = (batch_size, self.n_units)
        if self.h is None:
            self.set_state(np.zeros(state_shape, dtype=x.dtype))

        # run the cell
        h = self.h[:batch_size] # handle variable length seq batches (sorted descending)
        xh = ch.functions.hstack([x, h])
        r = sigmoid(matmul(xh, self.WU_r) + bcast_to(self.b_r, state_shape))
        z = sigmoid(matmul(xh, self.WU_z) + bcast_to(self.b_z, state_shape))
        x_rh = ch.functions.hstack([x, r * h])
        hbar = tanh(matmul(x_rh, self.WU_h) + bcast_to(self.b_h, state_shape))
        h = (1.-z)*h + z*hbar

        self.h[:batch_size] # carry over state

        return h

class BidirectionalGRU(ch.Link):
    """ A stateful Bidirectional GRU implementation.

    Given input vector :math:`x_f`, `x_b`, BidirectionalGRU returns the next
    hidden vectors :math:`h_f'`, `h_b'` defined as (wlog)

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    This is implemented as one link for speed and ease of handling variable length seqs.

    Since on the backward pass, the hidden state increases in size,
    the regular lstm/gru implementations can't handle them.

    However if we just feed in the sequences in reverse as a second input,
    then we can concatenate them together and the hidden state is a fixed size
    throughout the sequence.

    NOTE: n_units are the number of units for each cell. So the output is double this
    """
    def __init__(self, n_units, n_inputs=None,
                 W_init=None, U_init=None, bias_init=None):
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(BidirectionalGRU, self).__init__(
            # forward gru
            f_W_r=(n_inputs, n_units),
            f_U_r=(n_units, n_units),
            f_b_r=(n_units,),
            f_W_z=(n_inputs, n_units),
            f_U_z=(n_units, n_units),
            f_b_z=(n_units,),
            f_W_h=(n_inputs, n_units),
            f_U_h=(n_units, n_units),
            f_b_h=(n_units,),
            # backward gru
            b_W_r=(n_inputs, n_units),
            b_U_r=(n_units, n_units),
            b_b_r=(n_units,),
            b_W_z=(n_inputs, n_units),
            b_U_z=(n_units, n_units),
            b_b_z=(n_units,),
            b_W_h=(n_inputs, n_units),
            b_U_h=(n_units, n_units),
            b_b_h=(n_units,)
        )
        # initialize params
        if not W_init:
            W_init = ch.initializers.GlorotUniform(scale=1.0)
        W_init(self.f_W_r.data)
        W_init(self.f_W_z.data)
        W_init(self.f_W_h.data)
        W_init(self.b_W_r.data)
        W_init(self.b_W_z.data)
        W_init(self.b_W_h.data)
        if not U_init:
            U_init = ch.initializers.Orthogonal(scale=1.0)
        U_init(self.f_U_r.data)
        U_init(self.f_U_z.data)
        U_init(self.f_U_h.data)
        U_init(self.b_U_r.data)
        U_init(self.b_U_z.data)
        U_init(self.b_U_h.data)
        if not bias_init:
            bias_init = ch.initializers.Zero()
        bias_init(self.f_b_r.data)
        bias_init(self.f_b_z.data)
        bias_init(self.f_b_h.data)
        bias_init(self.b_b_r.data)
        bias_init(self.b_b_z.data)
        bias_init(self.b_b_h.data)

        # concat W's and U's for matmul speed
        self.f_WU_r = ch.functions.vstack([self.f_W_r, self.f_U_r])
        self.f_WU_z = ch.functions.vstack([self.f_W_z, self.f_U_z])
        self.f_WU_h = ch.functions.vstack([self.f_W_h, self.f_U_h])
        self.b_WU_r = ch.functions.vstack([self.b_W_r, self.b_U_r])
        self.b_WU_z = ch.functions.vstack([self.b_W_z, self.b_U_z])
        self.b_WU_h = ch.functions.vstack([self.b_W_h, self.b_U_h])

        # concat fwd and bwd cells for matmul speed
        self.WU_r = ch.functions.hstack([self.f_WU_r, self.b_WU_r])
        self.WU_z = ch.functions.hstack([self.f_WU_z, self.b_WU_z])
        self.WU_h = ch.functions.hstack([self.f_WU_h, self.b_WU_h])

        self.reset_state()

    def set_state(self, f_array, b_array):
        self.h_f = ch.Variable(f_array)
        self.h_b = ch.Variable(b_array)

    def reset_state(self):
        self.h_f = None
        self.h_b = None

    def __call__(self, x_f, x_b):
        # function shorthand
        sigmoid = ch.functions.sigmoid
        tanh = ch.functions.tanh
        matmul = ch.functions.matmul
        hstack = ch.functions.hstack
        bcast_to = ch.functions.broadcast_to

        # set the state if needed
        f_batch_size = x_f.shape[0]
        b_batch_size = x_b.shape[0]
        f_state_shape = (f_batch_size, self.n_units)
        b_state_shape = (b_batch_size, self.n_units)
        if self.h_f is None:
            self.set_state(np.zeros((f_batch_size, self.n_units), dtype=x_f.dtype),
                           np.zeros((f_batch_size, self.n_units), dtype=x_b.dtype))

        # run the cell
        # handle variable length seq batches (sorted descending)
        # NOTE that we don't need to set self.h at the end (it happens automatically)
        # since h is only a reference to self.h.data in memory
        # fwd cell
        h_f = self.h_f[:f_batch_size]
        xh_f = ch.functions.hstack([x_f, h_f])
        r_f = sigmoid(matmul(xh_f, self.f_WU_r) + bcast_to(self.f_b_r, f_state_shape))
        z_f = sigmoid(matmul(xh_f, self.f_WU_z) + bcast_to(self.f_b_z, f_state_shape))
        x_rh_f = ch.functions.hstack([x_f, r_f * h_f])
        hbar_f = tanh(matmul(x_rh_f, self.f_WU_h) + bcast_to(self.f_b_h, f_state_shape))
        h_f = (1.-z_f)*h_f + z_f*hbar_f
        # bkwd cell
        h_b = self.h_b[:b_batch_size]
        xh_b = ch.functions.hstack([x_b, h_b])
        r_b = sigmoid(matmul(xh_b, self.b_WU_r) + bcast_to(self.b_b_r, b_state_shape))
        z_b = sigmoid(matmul(xh_b, self.b_WU_z) + bcast_to(self.b_b_z, b_state_shape))
        x_rh_b = ch.functions.hstack([x_b, r_b * h_b])
        hbar_b = tanh(matmul(x_rh_b, self.b_WU_h) + bcast_to(self.b_b_h, b_state_shape))
        h_b = (1.-z_b)*h_b + z_b*hbar_b

        return h_f, h_b
