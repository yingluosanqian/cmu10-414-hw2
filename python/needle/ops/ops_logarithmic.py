from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_Z2 = array_api.max(Z, axis=self.axes)
        res = array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes)) + max_Z2
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node:Tensor):
        ### BEGIN YOUR SOLUTION
        max_Z = node.inputs[0].realize_cached_data().max(axis=self.axes, keepdims=True)
        exp_Z = exp(node.inputs[0] - max_Z)
        softmax_Z = exp_Z / summation(exp_Z, axes=self.axes, keepdims=True).broadcast_to(exp_Z.shape)
        if self.axes is None:
            return softmax_Z * out_grad.broadcast_to(node.inputs[0].shape)
        else:
            return softmax_Z * out_grad.reshape(max_Z.shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

