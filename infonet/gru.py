""" An adaptation of chainer's GRU that can handle variable length sequences2arrays.

Pretty much a complete rewrite.

http://docs.chainer.org/en/stable/_modules/chainer/links/connection/gru.html#StatefulGRU
"""
import numpy as np
import chainer as ch
import chainer.functions as F

from dropout import dropout

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
                #  W_init=None, U_init=None,
                 WU_init=None,
                 bias_init=None,
                 dropout=0.):
        self.dropout = dropout
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(GRU, self).__init__(
            # W_r=(n_inputs, n_units),
            # U_r=(n_units, n_units),
            # b_r=(n_units,),
            # W_z=(n_inputs, n_units),
            # U_z=(n_units, n_units),
            # b_z=(n_units,),
            # W_h=(n_inputs, n_units),
            # U_h=(n_units, n_units),
            # b_h=(n_units,)
            WU_r=(n_inputs+n_units, n_units),
            b_r =(n_units,),
            WU_z=(n_inputs+n_units, n_units),
            b_z =(n_units,),
            WU_h=(n_inputs+n_units, n_units),
            b_h =(n_units,),
        )
        # initialize params
        # if not W_init:
        #     W_init = ch.initializers.GlorotUniform(scale=1.0)
        # W_init(self.W_r.data)
        # W_init(self.W_z.data)
        # W_init(self.W_h.data)
        # if not U_init:
        #     U_init = ch.initializers.Orthogonal(scale=1.0)
        # U_init(self.U_r.data)
        # U_init(self.U_z.data)
        # U_init(self.U_h.data)
        if not WU_init:
            WU_init = ch.initializers.GlorotUniform()
        WU_init(self.WU_r.data)
        WU_init(self.WU_z.data)
        WU_init(self.WU_h.data)
        if not bias_init:
            bias_init = ch.initializers.Zero()
        bias_init(self.b_r.data)
        bias_init(self.b_z.data)
        bias_init(self.b_h.data)

        self.reset_state()

    def set_state(self, array):
        self.h = ch.Variable(array)
        if self.dropout:
            self.h, self.h_drop_mask = dropout(self.h, self.dropout, return_mask=True)

    def reset_state(self):
        self.h = None
        self.h_drop_mask = None

    def __call__(self, x, train=True):
        # set the state to zeros if no prior state
        batch_size = x.shape[0]
        state_shape = (batch_size, self.n_units)
        if self.h is None:
            self.set_state(np.zeros(state_shape, dtype=x.dtype))

        # split off variable len seqs
        if self.h.shape[0] > batch_size:
            h, h_rest = F.split_axis(self.h, [batch_size], 0)
            if self.h_drop_mask is not None:
                h_drop_mask = self.h_drop_mask[:batch_size]
        else:
            h, h_rest = self.h, None
            if self.h_drop_mask is not None:
                h_drop_mask = self.h_drop_mask

        # apply horizontal dropout
        # h = F.dropout(h, self.dropout, train=train)
        if self.h_drop_mask is not None:
            h = dropout(h, self.dropout, train=train, mask=h_drop_mask)
            # print 'self.h', self.h.shape, self.h.size, np.sum(np.isnan(self.h.data))
            # print 'h ', h.shape, h.size, np.sum(np.isnan(h.data)), np.argwhere(np.isnan(h.data))[:3]
        # run the cell
        xh = F.hstack([x, h])
        r = F.sigmoid(F.bias(F.matmul(xh, self.WU_r), self.b_r))
        z = F.sigmoid(F.bias(F.matmul(xh, self.WU_z), self.b_z))
        x_rh = F.hstack([x, r * h])
        hbar = F.tanh(F.bias(F.matmul(x_rh, self.WU_h), self.b_h))
        # NOTE: usually h is calculated without squashing, but this coupled with the repeated dropout
        # was causing steady explosion of h values due to repeated application of dropout scale factors
        if self.h_drop_mask is not None:
            h = F.tanh((1.-z)*h + z*hbar)
        else:
            h = (1.-z)*h + z*hbar

        # persist state
        if h_rest is not None:
            self.h = F.vstack([h, h_rest])
        else:
            self.h = h
        # self.h[:batch_size].data = h.data # carry over state
        # if train:
        #     self.h *= self.rnn_drop_mask

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
    and only update hidden states for seen seqs.

    NOTE: n_units are the number of units for each cell. So the output is double this
    """
    def __init__(self, n_units, n_inputs=None,
                #  W_init=None, U_init=None,
                 WU_init=None,
                 bias_init=None,
                 dropout=0.):
        self.dropout = dropout
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(BidirectionalGRU, self).__init__(
            # forward gru
            # f_W_r=(n_inputs, n_units),
            # f_U_r=(n_units, n_units),
            f_WU_r=(n_inputs+n_units, n_units),
            f_b_r=(n_units,),
            # f_W_z=(n_inputs, n_units),
            # f_U_z=(n_units, n_units),
            f_WU_z=(n_inputs+n_units, n_units),
            f_b_z=(n_units,),
            # f_W_h=(n_inputs, n_units),
            # f_U_h=(n_units, n_units),
            f_WU_h=(n_inputs+n_units, n_units),
            f_b_h=(n_units,),
            # backward gru
            # b_W_r=(n_inputs, n_units),
            # b_U_r=(n_units, n_units),
            b_WU_r=(n_inputs+n_units, n_units),
            b_b_r=(n_units,),
            # b_W_z=(n_inputs, n_units),
            # b_U_z=(n_units, n_units),
            b_WU_z=(n_inputs+n_units, n_units),
            b_b_z=(n_units,),
            # b_W_h=(n_inputs, n_units),
            # b_U_h=(n_units, n_units),
            b_WU_h=(n_inputs+n_units, n_units),
            b_b_h=(n_units,)
        )
        # initialize params
        # if not W_init:
        #     W_init = ch.initializers.GlorotUniform(scale=1.0)
        # W_init(self.f_W_r.data)
        # W_init(self.f_W_z.data)
        # W_init(self.f_W_h.data)
        # W_init(self.b_W_r.data)
        # W_init(self.b_W_z.data)
        # W_init(self.b_W_h.data)
        # if not U_init:
        #     U_init = ch.initializers.Orthogonal(scale=1.0)
        # U_init(self.f_U_r.data)
        # U_init(self.f_U_z.data)
        # U_init(self.f_U_h.data)
        # U_init(self.b_U_r.data)
        # U_init(self.b_U_z.data)
        # U_init(self.b_U_h.data)
        if not WU_init:
            WU_init = ch.initializers.GlorotUniform()
        WU_init(self.f_WU_r.data)
        WU_init(self.f_WU_z.data)
        WU_init(self.f_WU_h.data)
        WU_init(self.b_WU_r.data)
        WU_init(self.b_WU_z.data)
        WU_init(self.b_WU_h.data)
        if not bias_init:
            bias_init = ch.initializers.Zero()
        bias_init(self.f_b_r.data)
        bias_init(self.f_b_z.data)
        bias_init(self.f_b_h.data)
        bias_init(self.b_b_r.data)
        bias_init(self.b_b_z.data)
        bias_init(self.b_b_h.data)

        # concat W's and U's for matmul speed
        # self.add_persistent('f_WU_r', ch.functions.vstack([self.f_W_r, self.f_U_r]))
        # self.add_persistent('f_WU_z', ch.functions.vstack([self.f_W_z, self.f_U_z]))
        # self.add_persistent('f_WU_h', ch.functions.vstack([self.f_W_h, self.f_U_h]))
        # self.add_persistent('b_WU_r', ch.functions.vstack([self.b_W_r, self.b_U_r]))
        # self.add_persistent('b_WU_z', ch.functions.vstack([self.b_W_z, self.b_U_z]))
        # self.add_persistent('b_WU_h', ch.functions.vstack([self.b_W_h, self.b_U_h]))

        # self.f_WU_r = ch.functions.vstack([self.f_W_r, self.f_U_r])
        # self.f_WU_z = ch.functions.vstack([self.f_W_z, self.f_U_z])
        # self.f_WU_h = ch.functions.vstack([self.f_W_h, self.f_U_h])
        # self.b_WU_r = ch.functions.vstack([self.b_W_r, self.b_U_r])
        # self.b_WU_z = ch.functions.vstack([self.b_W_z, self.b_U_z])
        # self.b_WU_h = ch.functions.vstack([self.b_W_h, self.b_U_h])
        # self.add_persistant('f_WU_r', )

        # # concat fwd and bwd cells for matmul speed
        # self.WU_r = ch.functions.hstack([self.f_WU_r, self.b_WU_r])
        # self.WU_z = ch.functions.hstack([self.f_WU_z, self.b_WU_z])
        # self.WU_h = ch.functions.hstack([self.f_WU_h, self.b_WU_h])

        self.reset_state()

    def set_state(self, f_array, b_array):
        self.h_f = ch.Variable(f_array)
        self.h_b = ch.Variable(b_array)
        # f_scale = f_array.dtype.type(1. / (1 - self.dropout))
        # f_flag = np.random.rand(*f_array.shape) >= self.dropout
        # self.f_rnn_drop_mask = ch.Variable(f_scale * f_flag)
        # b_scale = b_array.dtype.type(1. / (1 - self.dropout))
        # b_flag = np.random.rand(*b_array.shape) >= self.dropout
        # self.b_rnn_drop_mask = ch.Variable(b_scale * b_flag)
        # print self.b_rnn_drop_mask.shape

    def reset_state(self):
        self.h_f = None
        self.h_b = None
        # self.f_rnn_drop_mask = None
        # self.b_rnn_drop_mask = None

    def __call__(self, x_f, x_b, train=True):
        # function shorthand
        # sigmoid = ch.functions.sigmoid
        # # sigmoid = ch.functions.hard_sigmoid
        # tanh = ch.functions.tanh
        # matmul = ch.functions.matmul
        # hstack = ch.functions.hstack
        # bcast_to = ch.functions.broadcast_to

        # set the state if needed
        f_batch_size = x_f.shape[0]
        b_batch_size = x_b.shape[0]
        f_state_shape = (f_batch_size, self.n_units)
        b_state_shape = (b_batch_size, self.n_units)
        if self.h_f is None:
            self.set_state(np.zeros((f_batch_size, self.n_units), dtype=x_f.dtype),
                           np.zeros((f_batch_size, self.n_units), dtype=x_b.dtype))

        # split off variable len seqs
        if self.h_f.shape[0] > f_batch_size:
            h_f, h_f_rest = F.split_axis(self.h_f, [f_batch_size], 0)
            # if self.h_drop_mask is not None:
            #     h_drop_mask = self.h_drop_mask[:batch_size]
        else:
            h_f, h_f_rest = self.h_f, None
            # if self.h_drop_mask is not None:
            #     h_drop_mask = self.h_drop_mask
        if self.h_b.shape[0] > b_batch_size:
            h_b, h_b_rest = F.split_axis(self.h_b, [b_batch_size], 0)
            # if self.h_drop_mask is not None:
            #     h_drop_mask = self.h_drop_mask[:batch_size]
        else:
            h_b, h_b_rest = self.h_b, None
            # if self.h_drop_mask is not None:
            #     h_drop_mask = self.h_drop_mask

        # concat W's and U's for matmul speed
        # self.f_WU_r = ch.functions.vstack([self.f_W_r, self.f_U_r])
        # self.f_WU_z = ch.functions.vstack([self.f_W_z, self.f_U_z])
        # self.f_WU_h = ch.functions.vstack([self.f_W_h, self.f_U_h])
        # self.b_WU_r = ch.functions.vstack([self.b_W_r, self.b_U_r])
        # self.b_WU_z = ch.functions.vstack([self.b_W_z, self.b_U_z])
        # self.b_WU_h = ch.functions.vstack([self.b_W_h, self.b_U_h])

        # fwd cell
        # if train:
        #     self.h_f = self.h_f * self.f_rnn_drop_mask
        # h_f = self.h_f[:f_batch_size]
        # if train:
        #     h_f *= self.f_rnn_drop_mask[:f_batch_size]
        xh_f = F.hstack([x_f, h_f])
        r_f = F.sigmoid(F.bias(F.matmul(xh_f, self.f_WU_r), self.f_b_r))
        z_f = F.sigmoid(F.bias(F.matmul(xh_f, self.f_WU_z), self.f_b_z))
        x_rh_f = F.hstack([x_f, r_f * h_f])
        hbar_f = F.tanh(F.bias(F.matmul(x_rh_f, self.f_WU_h), self.f_b_h))
        h_f = (1.-z_f)*h_f + z_f*hbar_f
        # self.h_f.data[:f_batch_size] = h_f.data


        # bkwd cell
        xh_b = F.hstack([x_b, h_b])
        r_b = F.sigmoid(F.bias(F.matmul(xh_b, self.b_WU_r), self.b_b_r))
        z_b = F.sigmoid(F.bias(F.matmul(xh_b, self.b_WU_z), self.b_b_z))
        x_rh_b = F.hstack([x_b, r_b * h_b])
        hbar_b = F.tanh(F.bias(F.matmul(x_rh_b, self.b_WU_h), self.b_b_h))
        h_b = (1.-z_b)*h_b + z_b*hbar_b
        # if train:
        #     self.h_b = self.h_b * self.b_rnn_drop_mask
        # h_b = self.h_b[:b_batch_size]
        # # if train:
        # #     h_b *= self.b_rnn_drop_mask[:b_batch_size]
        # xh_b = ch.functions.hstack([x_b, h_b])
        # r_b = sigmoid(matmul(xh_b, self.b_WU_r) + bcast_to(self.b_b_r, b_state_shape))
        # z_b = sigmoid(matmul(xh_b, self.b_WU_z) + bcast_to(self.b_b_z, b_state_shape))
        # x_rh_b = ch.functions.hstack([x_b, r_b * h_b])
        # hbar_b = tanh(matmul(x_rh_b, self.b_WU_h) + bcast_to(self.b_b_h, b_state_shape))
        # h_b = (1.-z_b)*h_b + z_b*hbar_b
        # self.h_b.data[:b_batch_size] = h_b.data


        # persist state
        if h_f_rest is not None:
            self.h_f = F.vstack([h_f, h_f_rest])
        else:
            self.h_f = h_f
        if h_b_rest is not None:
            self.h_b = F.vstack([h_b, h_b_rest])
        else:
            self.h_b = h_b

        return h_f, h_b
