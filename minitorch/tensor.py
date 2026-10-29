"""
Implementation of the core Tensor object for autodifferentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Cast,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
    tensor_from_numpy,
    zeros_tensor_from_numpy,
    PowerScalar,
    Tanh,
    Attn_Softmax,
    LayerNorm,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


datatype = np.float32

@dataclass
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend
        if backend.device_storage:
            from .cuda_storage import DeviceStorage
            if not isinstance(v._storage, DeviceStorage):
                v._storage = DeviceStorage.from_numpy(v._storage, allocator=getattr(backend, 'memory_pool', None))


    # def requires_grad_(self, x: bool) -> None:
    #     self.history = History()

    def requires_grad_(self, x: bool) -> None:
        self.history = History() if x else None

    def requires_grad(self) -> bool:
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[datatype]:
        """
        Returns:
             Converted to numpy array
        """
        storage = self.detach().contiguous()._tensor._storage
        if getattr(storage, "is_device_storage", False):
            storage = storage.to_numpy()
        elif hasattr(storage, "copy_to_host"):
            storage = storage.copy_to_host()
        return np.array(storage, copy=True).reshape(self.shape)

    def to(self, backend=None, *, dtype=None):
        """Differentiable same-backend cast; backend transfers remain detached.

        Host backends support FP32 only. An implicit device-to-host transfer
        exports low precision values as FP32; explicit host half requests fail.
        """
        from .dtypes import validate_backend_dtype
        backend = self.backend if backend is None else backend
        target = dtype if dtype is not None else (self.dtype if backend.device_storage else 'float32')
        target = validate_backend_dtype(backend, target)
        if backend is self.backend:
            return self if target == self.dtype else Cast.apply(self, target)
        from .autodiff import is_grad_enabled
        return tensor_from_numpy(self.to_numpy(), backend=backend,
                                 requires_grad=is_grad_enabled() and self.requires_grad(), dtype=target)

    def cpu(self):
        from .tensor_ops import SimpleBackend
        return self.to(SimpleBackend)

    # Properties
    @property
    def dtype(self):
        from .dtypes import normalize_dtype
        return normalize_dtype(self._tensor._storage.dtype)

    @property
    def itemsize(self):
        from .dtypes import itemsize
        return itemsize(self.dtype)

    @property
    def nbytes(self):
        return self.size * self.itemsize

    @property
    def shape(self) -> UserShape:
        """
        Returns:
             shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """
        Returns:
             int : size of the tensor
        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """
        Returns:
             int : dimensionality of the tensor
        """
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        "Turns a python number into a tensor with the same backend."
        if isinstance(b, (int, float)):
            if self.backend.device_storage:
                key = (self.dtype, float(b))
                if key not in self.backend._constants:
                    if len(self.backend._constants) >= 128:
                        self.backend._constants.pop(next(iter(self.backend._constants)))
                    self.backend._constants[key] = self.backend.ops.full((1,), float(b), self.backend, self.dtype)
                c = self.backend._constants[key]
            else:
                c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        from .amp import autocast_dtype
        dtype = autocast_dtype()
        if dtype is not None:
            return MatMul.apply(self.to(dtype=dtype), b.to(dtype=dtype))
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b
    
    def __pow__(self, b: TensorLike) -> Tensor: 
        if isinstance(b, (int, float)):
            return PowerScalar.apply(self, self._ensure_tensor(b))
        elif len(b.shape) == 1:
            return PowerScalar.apply(self, self._ensure_tensor(b))
        else:
            raise NotImplementedError

    def all(self, dim: Optional[int] = None) -> Tensor:
        if dim is None:
            return All.apply(self.contiguous().view(self.size), tensor([0]))
        else:
            return All.apply(self, dim if isinstance(dim, Tensor) else tensor([dim]))

    def is_close(self, y: Tensor) -> Tensor:
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return ReLU.apply(self)

    def log(self) -> Tensor:
        return Log.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)
    
    def tanh(self) -> Tensor:
        return Tanh.apply(self)

    def item(self) -> float:
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def sum(self, dim: Optional[int] = None) -> Tensor:
        "Compute the sum over dimension `dim`"
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), tensor([0]))
        else:
            return Sum.apply(self, dim if isinstance(dim, Tensor) else tensor([dim]))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        "Compute the mean over dimension `dim`"
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size
    
    def var(self, dim: Optional[int] = None) -> Tensor:
        "Compute the variance over dimension `dim`"
        if dim is not None:
            shape = self.shape
            
            mean = self.sum(dim) / self.shape[dim]
            # You don't need to view again because it'll be the correct shape for broadcasting
            mean = mean.contiguous()
            
            diff = self.__sub__(mean) ** 2
            diff = diff.sum(dim) / self.shape[dim]
            
            return diff
        else:
            shape = self.shape
            mean = self.sum() / self.size
            mean = mean.contiguous().view(shape)
            
            diff = self.__sub__(mean) ** 2
            diff = diff.sum() / self.size
            
            return diff

    def permute(self, *order: int) -> Tensor:
        "Permute tensor dimensions to *order"
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        "Change the shape of the tensor to a new shape with the same size"
        return View.apply(self, tensor(list(shape)))

    def contiguous(self) -> Tensor:
        "Return a contiguous tensor with the same data"
        return self if self._tensor.is_contiguous() else Copy.apply(self)

    def clone(self) -> Tensor:
        "Return an independent contiguous copy, even if already contiguous."
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        if backend is self.backend:
            if backend.cuda:
                self._tensor.to_cuda_()
            return
        if self.backend.device_storage or backend.device_storage:
            raise ValueError("Use tensor.to(backend) for an explicit device transfer")
        self.backend = backend
        self.f = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        "Create a new tensor from data"
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """
        Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Parameters:
            other : backward tensor (must broadcast with self)

        Returns:
            Expanded version of `other` with the right derivatives

        """

        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        # Do not narrow FP32 GEMM gradients before broadcast reduction.
        buf = zeros_tensor_from_numpy(true_shape, self.backend, dtype=other.dtype)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        def zero(shape: UserShape) -> Tensor:
            # return Tensor.make(
            #     [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            # )
            return zeros_tensor_from_numpy(shape, self.backend, dtype=self.dtype)

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x : value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        x = x.to(dtype=self.dtype)
        buffer = getattr(self, '_grad_buffer', None)
        if buffer is not None:
            if buffer._tensor._storage.owner is x._tensor._storage.owner:
                x = x.clone()
            if self.grad is None:
                self.backend.id_map(x, buffer)
            else:
                if self.grad is not buffer:
                    self.backend.id_map(self.grad, buffer)
                self.backend.ops.accumulate_into(buffer, x)
            self.grad = buffer
            return
        if self.grad is None:
            self.grad = self.zeros()
        self.grad += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
            if not inp.is_constant()
        ]

    def backward(self, grad_output: Optional[Tensor] = None, retain_graph: Optional[bool] = None) -> None:
        # Preserve legacy repeated-backward behavior except in opt-in planned mode.
        if retain_graph is None:
            retain_graph = getattr(self.backend, 'memory_mode', 'default') != 'planned'
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = self._ensure_tensor(1.0)
        backpropagate(self, grad_output, retain_graph=retain_graph)

    def release_history(self):
        if self.history is not None and self.history.last_fn is not None:
            self.history.ctx.release()
            self.history.inputs = ()

    def zero_grad_(self) -> None:  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.grad = None

    def attn_softmax(self, mask: Tensor) -> Tensor:
        from .amp import autocast_dtype
        dtype = autocast_dtype()
        inp = self.to(dtype=dtype) if dtype is not None else self
        return Attn_Softmax.apply(inp, mask.to(dtype='float32'))

    def layernorm(self, gamma: Tensor, beta: Tensor) -> Tensor:
        from .amp import autocast_dtype
        dtype = autocast_dtype()
        inp = self.to(dtype=dtype) if dtype is not None else self
        return LayerNorm.apply(inp, gamma.to(dtype='float32'), beta.to(dtype='float32'))
