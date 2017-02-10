""" An adaptation of chainer's GRU that can handle variable length sequences2arrays.

Pretty much a complete rewrite.

TODO: Instead of rescaling activations during training with dropout, which has numerical instabilities,
instead rescale the weights after training by the same factors.

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
                 WU_init=None,
                 bias_init=None,
                 dropout=0.0,
                 hdropout=0.0):
        self.dropout = dropout
        self.hdropout = hdropout
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(GRU, self).__init__(
            WU_r=(n_inputs+n_units, n_units),
            b_r =(n_units,),
            WU_z=(n_inputs+n_units, n_units),
            b_z =(n_units,),
            WU_h=(n_inputs+n_units, n_units),
            b_h =(n_units,),
        )
        # initialize params
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

    def set_state(self, array, train=True):
        self.h = ch.Variable(array)
        if self.hdropout:
            self.h_mask = np.random.rand(*array.shape) > self.hdropout
            if train:
                self.h *= self.h_mask
        else:
            self.h_mask = None

    def reset_state(self):
        self.h = None
        self.h_mask = None

    def rescale_Us(self):
        """ Rescale Us so activation are same in expectation at test time. """
        if self.hdropout:
            rate = 1./(1-self.hdropout)
            self.WU_r.data[self.n_inputs:,:] *= rate
            self.WU_z.data[self.n_inputs:,:] *= rate
            self.WU_h.data[self.n_inputs:,:] *= rate

    def __call__(self, x, train=True):
        # set the state to zeros if no prior state
        batch_size = x.shape[0]
        state_shape = (batch_size, self.n_units)
        if self.h is None:
            self.set_state(np.zeros(state_shape, dtype=x.dtype), train=train)

        # split off variable len seqs
        if self.h.shape[0] > batch_size:
            h, h_rest = F.split_axis(self.h, [batch_size], 0)
            if self.h_mask is not None:
                h_mask = self.h_mask[:batch_size]
        else:
            h, h_rest = self.h, None
            if self.h_mask is not None:
                h_mask = self.h_mask

        # run the cell
        xh = F.hstack([x, h])
        r = F.sigmoid(F.bias(F.matmul(xh, self.WU_r), self.b_r))
        z = F.sigmoid(F.bias(F.matmul(xh, self.WU_z), self.b_z))
        x_rh = F.hstack([x, r * h])
        hbar = F.tanh(F.bias(F.matmul(x_rh, self.WU_h), self.b_h))
        h = (1.-z)*h + z*hbar

        # if using horizontal dropout apply now so it's persisted
        if train and self.h_mask is not None:
            h *= h_mask

        # persist state
        if h_rest is not None:
            self.h = F.vstack([h, h_rest])
        else:
            self.h = h

        # apply regular dropout if not using hdropout
        if train and self.h_mask is None and self.dropout:
            h = F.dropout(h, self.dropout, train=train)
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
                 WU_init=None,
                 bias_init=None,
                 dropout=0.0,
                 hdropout=0.0):
        self.dropout = dropout
        self.hdropout = hdropout
        if n_inputs is None:
            n_inputs = n_units
        self.n_inputs = n_inputs
        self.n_units = n_units
        super(BidirectionalGRU, self).__init__(
            # forward gru
            f_WU_r=(n_inputs+n_units, n_units),
            f_b_r=(n_units,),
            f_WU_z=(n_inputs+n_units, n_units),
            f_b_z=(n_units,),
            f_WU_h=(n_inputs+n_units, n_units),
            f_b_h=(n_units,),
            # backward gru
            b_WU_r=(n_inputs+n_units, n_units),
            b_b_r=(n_units,),
            b_WU_z=(n_inputs+n_units, n_units),
            b_b_z=(n_units,),
            b_WU_h=(n_inputs+n_units, n_units),
            b_b_h=(n_units,)
        )
        # initialize params
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

        self.reset_state()

    def set_state(self, f_array, b_array, train=True):
        self.h_f = ch.Variable(f_array)
        self.h_b = ch.Variable(b_array)
        if self.hdropout:
            self.h_mask_f = np.random.rand(*array.shape) > self.hdropout
            self.h_mask_b = np.random.rand(*array.shape) > self.hdropout
            if train:
                self.h_f *= self.h_mask_f
                self.h_b *= self.h_mask_b
        else:
            self.h_mask_f = None
            self.h_mask_b = None

    def reset_state(self):
        self.h_f = None
        self.h_b = None

    def rescale_Us(self):
        """ Rescale Us so activation are same in expectation at test time. """
        if self.hdropout:
            rate = 1./(1-self.hdropout)
            self.f_WU_r.data[self.n_inputs:,:] *= rate
            self.f_WU_z.data[self.n_inputs:,:] *= rate
            self.f_WU_h.data[self.n_inputs:,:] *= rate
            self.b_WU_r.data[self.n_inputs:,:] *= rate
            self.b_WU_z.data[self.n_inputs:,:] *= rate
            self.b_WU_h.data[self.n_inputs:,:] *= rate

    def __call__(self, x_f, x_b, train=True):
        # set the state if needed
        f_batch_size = x_f.shape[0]
        b_batch_size = x_b.shape[0]
        f_state_shape = (f_batch_size, self.n_units)
        b_state_shape = (b_batch_size, self.n_units)
        if self.h_f is None:
            self.set_state(np.zeros((f_batch_size, self.n_units), dtype=x_f.dtype),
                           np.zeros((f_batch_size, self.n_units), dtype=x_b.dtype),
                           train=train)

        # split off variable len seqs
        if self.h_f.shape[0] > f_batch_size:
            h_f, h_f_rest = F.split_axis(self.h_f, [f_batch_size], 0)
            if self.h_mask_f is not None:
                h_mask_f = self.h_mask_f[:f_batch_size]
        else:
            h_f, h_f_rest = self.h_f, None
            if self.h_mask_f is not None:
                h_mask_f = self.h_mask_f

        if self.h_b.shape[0] > b_batch_size:
            h_b, h_b_rest = F.split_axis(self.h_b, [b_batch_size], 0)
            if self.h_mask_b is not None:
                h_mask_b = self.h_mask_b[:b_batch_size]
        else:
            h_b, h_b_rest = self.h_b, None
            if self.h_mask_b is not None:
                h_mask_b = self.h_mask_b

        # forward cell
        xh_f = F.hstack([x_f, h_f])
        r_f = F.sigmoid(F.bias(F.matmul(xh_f, self.f_WU_r), self.f_b_r))
        z_f = F.sigmoid(F.bias(F.matmul(xh_f, self.f_WU_z), self.f_b_z))
        x_rh_f = F.hstack([x_f, r_f * h_f])
        hbar_f = F.tanh(F.bias(F.matmul(x_rh_f, self.f_WU_h), self.f_b_h))
        h_f = (1.-z_f)*h_f + z_f*hbar_f

        # backward cell
        xh_b = F.hstack([x_b, h_b])
        r_b = F.sigmoid(F.bias(F.matmul(xh_b, self.b_WU_r), self.b_b_r))
        z_b = F.sigmoid(F.bias(F.matmul(xh_b, self.b_WU_z), self.b_b_z))
        x_rh_b = F.hstack([x_b, r_b * h_b])
        hbar_b = F.tanh(F.bias(F.matmul(x_rh_b, self.b_WU_h), self.b_b_h))
        h_b = (1.-z_b)*h_b + z_b*hbar_b

        # if using horizontal dropout apply now so it's persisted
        if train and self.h_mask_f is not None:
            h_f *= h_mask_f
        if train and self.h_mask_b is not None:
            h_b *= h_mask_b

        # persist state
        if h_f_rest is not None:
            self.h_f = F.vstack([h_f, h_f_rest])
        else:
            self.h_f = h_f
        if h_b_rest is not None:
            self.h_b = F.vstack([h_b, h_b_rest])
        else:
            self.h_b = h_b

        # apply regular dropout if not using hdropout
        if train and self.h_mask_f is None and self.dropout:
            h_f = F.dropout(h_f, self.dropout, train=train)
        if train and self.h_mask_b is None and self.dropout:
            h_b = F.dropout(h_b, self.dropout, train=train)
        return h_f, h_b

class StackedGRU(ch.Chain):
    """ Higher level GRU implementation that handles stacking, dropout,
    horizontal dropout, bidirectionality, and operates over entire sequences.
    """
    def __init__(self, in_size, state_sizes,
                 dropouts=[],
                 hdropouts=[],
                 bidirectional=False):
        super(StackedGRU, self).__init__()
        self.bidirectional = bidirectional
        if dropouts:
            assert len(state_sizes) == len(dropouts), "Provide per layer dropouts"
        else:
            dropouts = [0.0]*len(state_sizes)
        if hdropouts:
            assert len(state_sizes) == len(hdropouts), "Provide per layer dropouts"
        else:
            hdropouts = [0.0]*len(state_sizes)

        gru_params = zip(state_sizes, dropouts, hdropouts)
        prev_state_size = in_size
        self.grus = []
        if bidirectional:
            for i, (state_size, drop, hdrop) in enumerate(gru_params):
                gru = BidirectionalGRU(state_size//2,
                                       n_inputs=prev_state_size,
                                       dropout=drop, hdropout=hdrop)
                self.grus.append(gru)
                self.add_link('gru_'+str(i), gru)
                prev_state_size = state_size
        else:
            for i, (state_size, drop, hdrop) in enumerate(gru_params):
                gru = GRU(state_size,
                          n_inputs=prev_state_size,
                          dropout=drop, hdropout=hdrop)
                self.grus.append(gru)
                self.add_link('gru_'+str(i), gru)
                prev_state_size = state_size

    def __call__(self, xs, train=False):
        """ Operates on entire sequences """
        if self.bidirectional:
            fowards, backwards = [], []
            for h_f, h_b in zip(xs, xs[::-1]):
                for gru in self.grus:
                    h_f, h_b = gru(h_f, h_b, train=train)
                forwards.append(h_f)
                backwards.append(h_b)
            return [ F.hstack([h_f, h_b]) for h_f, h_b in zip(forwards, backwards)]
        else:
            hs = []
            for h in xs:
                for gru in self.grus:
                    h = gru(h, train=train)
                hs.append(h)
            return hs

    def reset_state(self):
        for gru in self.grus:
            gru.reset_state()

    def rescale_Us(self):
        """ Rescale Us so activation are same in expectation at test time. """
        for gru in self.grus:
            gru.rescale_Us()
