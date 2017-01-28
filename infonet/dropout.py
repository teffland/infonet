""" Minor change to dropout that allows for providing the mask """
import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio, mask=None, return_mask=False):
        self.dropout_ratio = dropout_ratio
        self.mask = mask
        if type(mask) is chainer.Variable: # handle incoming Variables
            self.mask = self.mask.data
        self.return_mask = return_mask

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if self.mask is None:
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            if xp == numpy:
                flag = xp.random.rand(*x[0].shape) >= self.dropout_ratio
            else:
                flag = (xp.random.rand(*x[0].shape, dtype=numpy.float32) >=
                        self.dropout_ratio)
            self.mask = scale * flag
        if self.return_mask:
            # print 'return mask', x[0], self.mask
            return x[0] * self.mask, self.mask
        else:
            # print 'not return mask', x[0], self.mask
            return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,


def dropout(x, ratio=.5, train=True, mask=None, return_mask=False):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If ``True``, executes dropout. Otherwise, does nothing.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.

    """
    if train:
        return Dropout(ratio, mask=mask, return_mask=return_mask)(x)
    return x
