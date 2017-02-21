""" Implements n-gram convolution for text with multipl n-gram sizes """

import chainer as ch
import chainer.links as L
import chainer.functions as F

class NGramConvolution(ch.Chain):
    def __init__(self, ngrams, n_filters, embed_size):
        super(NGramConvolution, self).__init__()
        if type(n_filters) in (tuple, list):
            assert len(n_filters) == len(ngrams)
        else:
            n_filters = [n_filters]*len(ngrams)
        self.convs = []
        for n, nf in zip(ngrams, n_filters):
            conv = L.Convolution2D(1, nf, ksize=(n, embed_size), pad=(n//2,0))
            self.add_link('conv_{}'.format(n), conv)
            self.convs.append(conv)
        self.ngrams = ngrams
        self.n_filters = n_filters
        self.out_size = sum(n_filters)

    def __call__(self, xs):
        if len(xs.shape) == 3: # [ batch size, words, embed size]
            xs = F.expand_dims(xs, 1)
        ys = []
        for conv in self.convs:
            y = conv(xs)
            y = F.transpose(F.squeeze(y, 3), [0,2,1])
            # if using even ngram, we will have one extra convolution,
            # since chainer doesn't support asymmetric padding
            # so we drop the final one
            if conv.ksize[0] % 2 == 0:
                y = y[:,:-1,:]
            # print y.shape
            ys.append(y)
        return F.dstack(ys)
