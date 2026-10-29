"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        weight_numpy = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        weight = tensor_from_numpy(weight_numpy, backend=backend)

        self.weights = Parameter(weight)
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        x_one_hot = one_hot(x, self.num_embeddings)
        x_ont_hot_2d = x_one_hot.contiguous().view(bs * seq_len, self.num_embeddings)
        output_2d = x_ont_hot_2d @ self.weights.value
        return output_2d.contiguous().view(bs, seq_len, self.embedding_dim)
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        ### BEGIN ASSIGN3_2
        if self.p_dropout == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.p_dropout
        if x.backend.device_storage:
            from .tensor_functions import rand
            mask = rand(x.shape, backend=x.backend) < keep_prob
            return x * mask / keep_prob
        mask = np.random.binomial(1, keep_prob, size=x.shape).astype(np.float32)

        mask = tensor_from_numpy(mask, backend=x.backend)
        return x * mask / keep_prob
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.in_size = in_size
        self.backend = backend

        bound = 1.0 / np.sqrt(in_size)
        def initial_random(shape):
            # Initialization is a one-time upload. Keep identical seeded weights
            # across host/device backends; runtime dropout uses device RNG.
            return rand(shape).to(backend) if backend.device_storage else rand(shape, backend=backend)
        weight = (initial_random((in_size, out_size)) * (2.0 * bound) - bound)
        
        self.weights = Parameter(weight)

        if bias:
            bias_value = (initial_random((out_size,)) * (2.0 * bound) - bound)
            self.bias = Parameter(bias_value)
        else:
            self.bias = None
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        output = x @ self.weights.value
        if self.bias is not None:
            from .amp import autocast_dtype
            bias = self.bias.value.to(dtype=output.dtype) if autocast_dtype() is not None else self.bias.value
            output += bias
        return output
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        self.layernorm_impl = "fused"
        ### BEGIN ASSIGN3_2
        self.backend = backend
        self.weights = Parameter(ones((self.dim,), backend=backend))
        self.bias = Parameter(zeros((self.dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        if self.layernorm_impl == "fused" and dim % 4 == 0 and dim <= 1024 and self.eps == 1e-5:
            return x.contiguous().layernorm(
                self.weights.value,
                self.bias.value,
            )

        mean = x.mean(dim=1)
        center = x - mean

        var = (center ** 2).mean(dim=1)

        normalized = center / ((var + self.eps) ** 0.5)
        output = normalized * self.weights.value + self.bias.value
        return output
        ### END ASSIGN3_2
