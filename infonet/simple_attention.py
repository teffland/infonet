""" Implements a simple attention mechanism for one sequence at a time """
import numpy as np
import chainer as ch
import chainer.functions as F
import chainer.links as L

from masked_softmax import masked_softmax

class BatchPaddedAttention(ch.Link):
    def __init__(self, n_dim):
        super(BatchPaddedAttention, self).__init__(
            w=(1, n_dim, 1)
        )
        ch.initializers.Normal()(self.w.data)

    def __call__(self, padded_xs, masks):
        """ Expects xs to be of shape [n, max_wid, h] """
        w = F.broadcast_to(self.w, (padded_xs.shape[0],)+self.w.shape[1:])
        scores = F.reshape(F.batch_matmul(padded_xs, w), padded_xs.shape[:2]) # [n x max wid]
        scores = masked_softmax(scores, masks[:,:,0])
        scores = F.broadcast_to(F.expand_dims(scores, 2), padded_xs.shape) # [ n x wid x h]
        scored = F.sum(scores * padded_xs, axis=1)
        return scored
