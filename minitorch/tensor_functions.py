"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import copy

import minitorch

from . import operators
from .autodiff import Context, is_grad_enabled
from .tensor_ops import SimpleBackend, TensorBackend
from .dtypes import validate_backend_dtype, promote_types

if TYPE_CHECKING:
    from typing import Any, List, Tuple, Union

    from .tensor import Tensor
    from .tensor_data import (UserIndex, UserShape,
                              Storage, OutIndex, Index, Shape, Strides)

datatype = np.float32
datasize = 4

def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if is_grad_enabled() and v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(b, grad_output),
            grad_output.f.mul_zip(a, grad_output),
        )


class PowerScalar(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, scalar: Tensor) -> Tensor:
        """Calculates the element-wise power of a to a single scalar.
        Equivalent to a ** scalar in numpy if a is a n-dimensional array and scalar is a scalar.

        Parameters
        ----------
            ctx : Context
                A context object you can temporarily store values to.
            a : Tensor
                The tensor to raise to the power of.
            scalar : Tensor
                The exponent of shape (1,).
        
        Returns
        -------
            output : Tensor
                Tensor containing the result of raising every element of a to scalar.
        """
        # COPY FROM ASSIGN3
        out = a.f.pow_scalar_zip(a, scalar)
        ctx.save_for_backward(a, scalar)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Calculates the gradient of the input a with respect to grad_output.
        NOTE: miniTorch requires that we two gradients: one for the input tensor and scalar.
        Technically, we should only return one gradient for the tensor since there is no gradient for a constant.
        
        Parameters
        ----------
            ctx : Context
                The same context used in forward.
            grad_output : Tensor
                The gradient in the backward pass with respect to the output of forward. (Same shape as forward's output.)
        
        Returns
        -------
            gradients : Tuple[Tensor, float]
                Tuple containing (gradient_for_a, 0.0)
                gradient_for_a must be the correct gradient, but just return 0.0 for the gradient of scalar.
        """
        a, scalar = ctx.saved_values
        grad_a    = None
        
        # COPY FROM ASSIGN3
        grad_a = grad_output * (scalar * (a ** (scalar - 1)))
        return (grad_a, 0.0)


class Tanh(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor: 
        """Calculates the element-wise tanh of a
        Equivalent to np.tanh(a) in numpy if a is a n-dimensional array.

        Parameters
        ----------
            ctx : Context
                A context object you can temporarily store values to.
            a : Tensor
                The tensor to take the tanh of.
        
        Returns
        -------
            output : Tensor
                Tensor containing the element-wise tanh of a.
        """
        # COPY FROM ASSIGN3
        out = a.f.tanh_map(a)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Calculates the gradient of the input a with respect to grad_output.
        
        Parameters
        ----------
            ctx : Context
                The same context used in forward.
            grad_output : Tensor
                The gradient in the backward pass with respect to the output of forward. (Same shape as forward's output.)
        
        Returns
        -------
            output : Tensor
                gradient_for_a must be the correct element-wise gradient for tanh.
        """
        # COPY FROM ASSIGN3
        out = ctx.saved_values[0]
        return grad_output * (-(out ** 2) + 1) 


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        out = t1.f.log_map(t1)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return grad_output.zeros(a_shape), grad_output.zeros(b_shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return grad_output.zeros(a_shape), grad_output.zeros(b_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        ctx.save_for_backward(order)
        return a._new(a._tensor.permute(*[int(order[i]) for i in range(order.size)]))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        order: Tensor = ctx.saved_values[0]
        order2: List[int] = [
            a[0]
            for a in sorted(
                enumerate([order[i] for i in range(order.size)]), key=lambda a: a[1]
            )
        ]
        return grad_output._new(grad_output._tensor.permute(*order2)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output.contiguous()._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class Cast(Function):
    @classmethod
    def apply(cls, a, dtype):
        dtype = validate_backend_dtype(a.backend, dtype)
        if a.dtype == dtype:
            return a
        need_grad = is_grad_enabled() and a.requires_grad()
        ctx = Context(not need_grad)
        ctx.save_for_backward(a.dtype)
        c = a.backend.ops.cast(a.detach(), dtype)
        back = minitorch.History(cls, ctx, (a,)) if need_grad else None
        return minitorch.Tensor(c._tensor, back, backend=c.backend)

    @staticmethod
    def backward(ctx, grad):
        (dtype,) = ctx.saved_values
        return grad.backend.ops.cast(grad, dtype)


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))
        dtype = promote_types(t1.dtype, t2.dtype)
        if t1.backend.device_storage and dtype != 'float32':
            # Operand dtype was recorded in forward, not taken from a current
            # autocast scope. FP32 outputs preserve parameter-gradient range.
            mm = t1.backend.ops.matrix_multiply
            return (mm(grad_output, transpose(t2), compute_dtype=dtype, output_dtype='float32'),
                    mm(transpose(t1), grad_output, compute_dtype=dtype, output_dtype='float32'))
        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


class FlashAttention(Function):
    @staticmethod
    def forward(ctx, q, k, v, query_lengths, key_lengths, causal):
        flag = bool(causal.item())
        out,lse,q,k,v,ql,kl = q.f.flash_attention_fw(q,k,v,query_lengths,key_lengths,flag)
        ctx.save_for_backward(q,k,v,out,lse,ql,kl,flag)
        return out

    @staticmethod
    def backward(ctx, grad):
        q,k,v,out,lse,ql,kl,flag = ctx.saved_values
        dq,dk,dv = q.f.flash_attention_bw(grad,q,k,v,out,lse,ql,kl,flag)
        return dq,dk,dv,0.0,0.0,0.0


class Attn_Softmax(Function):
    @staticmethod
    def forward(ctx: Context, inp: Tensor, mask: Tensor) -> Tensor:
      #   BEGIN ASSIGN4_1_1
      output = inp.f.attn_softmax_fw(inp, mask)
      ctx.save_for_backward(output)
      return output
      #   END ASSIGN4_1_1

    @staticmethod
    def backward(ctx: Context, out_grad: Tensor) -> Tensor:
      #   BEGIN ASSIGN4_1_2
      (soft_outp,) = ctx.saved_values

      inp_grad = soft_outp.f.attn_softmax_bw(out_grad, soft_outp)
      return inp_grad, 0.0
      #   END ASSIGN4_1_2


class LayerNorm(Function):
    @staticmethod
    def forward(ctx: Context, inp: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
      #   BEGIN ASSIGN4_2_1
      output, vars, means = inp.f.layernorm_fw(inp, gamma, beta)
      ctx.save_for_backward(inp, gamma, beta, vars, means)
      return output
      #   END ASSIGN4_2_1

    @staticmethod
    def backward(ctx: Context, out_grad: Tensor) -> Tensor:
      #   BEGIN ASSIGN4_2_2
        inp, gamma, beta, vars, means = ctx.saved_values
        inp_grad, gamma_grad, beta_grad = inp.f.layernorm_bw(
            out_grad, inp, gamma, beta, vars, means
        )
        return inp_grad, gamma_grad, beta_grad
      #   END ASSIGN4_2_2


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend, *, dtype='float32') -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return zeros_tensor_from_numpy(shape, backend, dtype=dtype)


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend, *, dtype='float32') -> Tensor:
    """
    Produce a ones tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    dtype = validate_backend_dtype(backend, dtype)
    if backend.device_storage:
        return backend.ops.full(shape, 1, backend, dtype)
    return minitorch.Tensor.make(
        [1] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
    *, dtype='float32',
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    dtype = validate_backend_dtype(backend, dtype)
    if backend.device_storage:
        result = backend.ops.random(shape, backend, random.getrandbits(64), dtype)
        result.requires_grad_(requires_grad)
        return result
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
    *, dtype='float32',
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    return tensor_from_numpy(np.asarray(ls, dtype=np.float32).reshape(shape),
                             backend, requires_grad, dtype=dtype)


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False,
    *, dtype='float32',
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, list) and len(ls) > 0 and (not isinstance(ls[0], list)):
            # Already flat
            return ls
        elif isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad, dtype=dtype)


def tensor_from_numpy(
    ls: Storage, backend: TensorBackend = SimpleBackend, requires_grad: bool = False,
    *, dtype='float32',
) -> Tensor:
    """NOTE: This should ONLY be used to initialize a tensor. 
    Any other usage could result in undefined behavior.
    """
    dtype = validate_backend_dtype(backend, dtype)
    if backend.device_storage:
        from .cuda_storage import DeviceStorage
        storage = DeviceStorage.from_numpy(ls, allocator=getattr(backend, 'memory_pool', None), dtype=dtype)
        res = minitorch.Tensor.make(storage, ls.shape, backend=backend)
        res.requires_grad_(requires_grad)
        return res
    if ls.dtype != datatype:
        ls = ls.astype(datatype)

    res =  minitorch.Tensor(
        v = minitorch.TensorData(
            ls.flatten(), # Will create a COPY of the numpy array
            ls.shape, 
            # flatten() is C-contiguous even when the source is a sliced view.
            None
        ),
        backend=backend
    )

    res.requires_grad_(requires_grad)
    
    return res


def zeros_tensor_from_numpy(shape, backend: TensorBackend = SimpleBackend, *, dtype='float32'):
    """NOTE: This should ONLY be used to initialize a tensor. 
    Any other usage could result in undefined behavior.
    """
    dtype = validate_backend_dtype(backend, dtype)
    if backend.device_storage:
        return backend.ops.full(shape, 0, backend, dtype)
    zs = np.zeros(shape, dtype=datatype)
    return minitorch.Tensor(
        v = minitorch.TensorData(
            zs.flatten(), # Will create a COPY of the numpy array
            shape, 
            tuple(i // datasize for i in zs.strides)
        ),
        backend=backend
    )


def ones_tensor_from_numpy(shape, backend: TensorBackend = SimpleBackend, *, dtype='float32'):
    """NOTE: This should ONLY be used to initialize a tensor. 
    Any other usage could result in undefined behavior.
    """
    dtype = validate_backend_dtype(backend, dtype)
    if backend.device_storage:
        return backend.ops.full(shape, 1, backend, dtype)
    zs = np.ones(shape).astype(datatype)
    return minitorch.Tensor(
        v = minitorch.TensorData(
            zs.flatten(), # Will create a COPY of the numpy array
            shape, 
            tuple(i // datasize for i in zs.strides)
        ),
        backend=backend
    )

# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    import torch
    x = vals[arg]
    up_np = np.zeros(x.shape, dtype=np.float64)
    up_np[ind] = epsilon
    vals1 = [torch.tensor(x.to_numpy().astype(np.float64)) if j != arg else torch.tensor(x.to_numpy().astype(np.float64) + up_np) for j, x in enumerate(vals)]
    vals2 = [torch.tensor(x.to_numpy().astype(np.float64)) if j != arg else torch.tensor(x.to_numpy().astype(np.float64) - up_np) for j, x in enumerate(vals)]
    delta = float(f(*vals1).sum() - f(*vals2).sum().numpy())
    # print(f"Debug in grad_central_difference: delta {delta}")
    return delta / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor, tol=1e-6) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """
    Gradient check error for function %s.

    Input %s

    Received derivative %f for argument %d and index %s,
    but was expecting derivative %f from central difference.

    """
    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
