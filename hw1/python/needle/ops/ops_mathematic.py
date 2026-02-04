"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # 与PowerScalar区分，此处b是NDArray
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # 注意此处所有操作都必须基于Tensor而不是NDArray
        grad_lhs = out_grad * rhs * (lhs ** (rhs - 1))
        grad_rhs = out_grad * (lhs ** rhs) * log(lhs)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.pow(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 注意这里是一元算子所以输入提取第一个，inputs方法的返回值是(input,)
        # 此处不能用input, _ = node.inputs，这会假设元组输出有两个元素
        input = node.inputs[0]
        return out_grad * self.scalar * (input ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        
        grad_lhs = out_grad / rhs
        grad_rhs = -out_grad * lhs * (rhs ** (-2))

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes=None):
        self.axes = axes

    def _get_axes(self, ndim):
        if self.axes is None:
            axes = list(range(ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
        elif len(self.axes) == 2:
            # 传入参数为交换axes的情况（2个参数）
            i, j = self.axes
            i %= ndim
            j %= ndim
            axes = list(range(ndim))
            axes[i], axes[j] = axes[j], axes[i]
        else:
            # 传入参数为完整序列的情况（2个以上参数）
            # 完整 permutation（backward会走到）
            axes = list(self.axes)
        return axes

    def compute(self, a):
        axes = self._get_axes(a.ndim)
        return array_api.transpose(a, tuple(axes))

    def gradient(self, out_grad, node):
        (inp,) = node.inputs
        ndim = len(inp.shape)
        axes = self._get_axes(ndim)

        inv_axes = [0] * ndim
        for i, ax in enumerate(axes):
            inv_axes[ax] = i

        return out_grad.transpose(tuple(inv_axes))


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        # 注意此处和transpose操作不同，无法通过再reshape一次还原
        # 而是要手动reshape为input的形状
        return out_grad.reshape(input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        # 广播操作一般作用于缺失或者大小为1的维度（从右往左对齐）
        # 广播操作的本质是某些元素被复制了很多次
        # x->y相当于是x的梯度被分给了多个y
        # backward就应该求和而不是平均
        input_shape = input.shape
        output_shape = out_grad.shape
        # 对齐维度（广播只能在左侧padding）
        ndim_diff = len(output_shape) - len(input_shape)
        # 下面的代码计算出padding后的形状（还未复制）
        padded_input_shape = (1,) * ndim_diff + input_shape

        axes = []
        for i, (in_dim, out_dim) in enumerate(zip(padded_input_shape, output_shape)):
            if in_dim == 1 and out_dim != 1:
                axes.append(i) # 添加padding的维度

        grad = out_grad
        if axes:
            grad = grad.sum(tuple(axes)) # 求和，注意此处要用tuple说明分别在这些axes上求和

        return grad.reshape(input_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        input_shape = input.shape
        # 错误示例：直接进行broadcast
        # 因为广播操作补充维度只允许left padding
        # 所以如果sum是在-1维度的话，广播是不合法的
        # return out_grad.broadcast_to(input_shape)
        if self.axes is None:
            return out_grad.broadcast_to(input_shape)
        # 所以先padding后广播
        # 计算reshape的中间形状
        axes = self.axes
        if isinstance(axes, int):
            axes = (axes,)

        shape = list(input_shape)
        for ax in axes:
            shape[ax] = 1

        # 进行广播操作
        return out_grad.reshape(shape).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # 错误示例1：只能处理二维情况
        # return out_grad @ rhs.T, lhs.T @ out_grad
        # 错误示例2：无法处理batch处理下进行了broadcast的情况
        # grad_lhs = matmul(out_grad, rhs.transpose((-1, -2)))
        # grad_rhs = matmul(lhs.transpose((-1, -2)), out_grad)
        # 判断是否进行了广播，如果进行了广播需要梯度求和
        grad_lhs = matmul(out_grad, rhs.transpose((-1, -2)))
        grad_rhs = matmul(lhs.transpose((-1, -2)), out_grad)
        if lhs.shape[:-2] != grad_lhs.shape[:-2]:
            grad_lhs = grad_lhs.sum(tuple(range(len(grad_lhs.shape) - len(lhs.shape))))
        if rhs.shape[:-2] != grad_rhs.shape[:-2]:
            grad_rhs = grad_rhs.sum(tuple(range(len(grad_rhs.shape) - len(rhs.shape))))
        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return out_grad / input
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
       return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

