""" Implements a simple attention mechanism for one sequence at a time """
import numpy as np
import chainer as ch
import chainer.functions as F
import chainer.links as L

class SimpleAttention(ch.Link):
    """ Implements an attention-weighted average.

    Note: Only works on one variable length sequence."""
    def __init__(self, n_dim):
        super(SimpleAttention, self).__init__(
            w=(1, n_dim)
        )
        ch.initializers.Normal()(self.w.data)

    def __call__(self, xs):
        """ Expects xs to be a list-like of n h-d vectors or of shape [n , h]"""
        if type(xs) in (tuple, list):
            xs = F.vstack(xs)
        scores = F.matmul(self.w, xs, transb=True)
        scores = F.reshape(F.softmax(scores), (-1,1)) # softmax only works on second axis
        scores = F.broadcast_to(scores, xs.shape)
        return F.sum(scores * xs, axis=0)
