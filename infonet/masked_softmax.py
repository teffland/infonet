""" Implementation of softmax operation that normalized over a specified support.

Adapted from chainers softmax
http://docs.chainer.org/en/stable/_modules/chainer/functions/activation/softmax.html#softmax

NOTE: This is only implemented for CPU
"""
import numpy as np

# from chainer import cuda
from chainer import function
from chainer.utils import type_check

# if cuda.cudnn_enabled:
#     cudnn = cuda.cudnn
#     libcudnn = cudnn.cudnn
#     _cudnn_version = libcudnn.getVersion()
#     _algorithm = libcudnn.CUDNN_SOFTMAX_ACCURATE
#     _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


class MaskedSoftmax(function.Function):

    """Softmax activation function that computes probabilities only over
       an input support mask.

    Eg, logit = [ -3, 4, 1], mask = [1,1,0]
    Then Z = sum(exp(-3) + exp(4))
    and
    probs = [ exp(-3)/Z, exp(4)/Z, 0.0]
    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, m_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim > 1,
        )

    def forward_cpu(self, inputs):
        x, mask = inputs
        y = x - x.max(axis=1, keepdims=True)
        self.y = np.exp(y) * mask
        self.y /= self.y.sum(axis=1, keepdims=True)

        return self.y,

    def backward_cpu(self, inputs, gy):
        x, mask = inputs
        gx = self.y * gy[0]
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= self.y * sumdx
        gx *= mask

        return gx,


def masked_softmax(x, mask):
    """Channelwise softmax function.

    This function computes its softmax along the second axis. Let
    :math:`x = (x_1, x_2, \\dots, x_d)^{\\top}` be the d dimensional index
    array and :math:`f(x)` be the d dimensional input array. For each index
    :math:`x` of the input array :math:`f(x)`, it computes the probability
    :math:`p(x)` defined as
    :math:`p(x) = {\\exp(f(x)) \\over \\sum_{x_2} \\exp(f(x))}`.

    Args:
        x (~chainer.Variable): Input variable.
        mask (bool, int, float): Mask representing support (along second axis)

    Returns:
        ~chainer.Variable: Output variable.

    """
    return MaskedSoftmax()(x, mask)
