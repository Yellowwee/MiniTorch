from typing import Tuple
import numba
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .cuda_ops import CudaOps
from .tensor import Tensor
from .tensor_functions import Function, FlashAttention, rand, tensor, tensor_from_numpy
import numpy as np
import math


def flash_attention(q, k, v, causal=False, dropout_p=0.0,
                    query_lengths=None, key_lengths=None):
    """FP32/FP16/BF16 tiled exact attention with FP32 accumulators.

    Causal alignment is top-left (key_index <= query_index). Lengths are [B]
    right-padding limits; empty rows produce zero output and zero gradients.
    No attention probabilities are returned or saved. Dropout is unsupported.
    Host length lists are validated/uploaded; device lengths must contain integer
    values in [0,N] and cannot require gradients. Inputs must be finite.
    """
    if dropout_p != 0:
        raise ValueError("FlashAttention currently requires dropout_p=0")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")
    if not q.backend.device_storage or k.backend is not q.backend or v.backend is not q.backend:
        raise ValueError("Q/K/V must use the same resident CUDA backend")
    from .amp import autocast_dtype
    dtype = autocast_dtype()
    if dtype is not None:
        q,k,v = (t.to(dtype=dtype) for t in (q,k,v))
    if not q.dtype == k.dtype == v.dtype:
        raise ValueError('FlashAttention Q/K/V must have the same dtype')
    if q.dims!=4 or k.dims!=4 or v.shape!=k.shape or q.shape[:2]!=k.shape[:2] or q.shape[3]!=k.shape[3]:
        raise ValueError("Expected Q[B,H,Nq,D], K/V[B,H,Nk,D]")
    b,h,nq,d = q.shape
    nk = k.shape[2]
    if min(b,h,nq,nk,d)<=0 or d>128 or b*h>65535:
        raise ValueError("FlashAttention requires positive dimensions, D<=128, B*H<=65535")
    if q.size>2**31-1 or k.size>2**31-1:
        raise ValueError("FlashAttention supports int32 indexing")
    def lengths(value,limit):
        if value is None:
            return q.backend.ops.full((b,),limit,q.backend)
        if isinstance(value,Tensor):
            if value.backend is not q.backend or value.shape!=(b,) or value.requires_grad() or value.dtype != 'float32':
                raise ValueError("Length tensors must be non-differentiable device tensors of shape [B]")
            return value
        a=np.asarray(value)
        if a.shape!=(b,) or not np.isfinite(a).all() or ((a<0)|(a>limit)|(a!=np.floor(a))).any():
            raise ValueError("Lengths must be integer values in [0, sequence length], shape [B]")
        return tensor_from_numpy(a,backend=q.backend)
    return FlashAttention.apply(q,k,v,lengths(query_lengths,nq),lengths(key_lengths,nk),tensor([int(causal)]))

def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_width = width // kw
    new_height = height // kh

    x = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(batch, channel, new_height, new_width, kh * kw)
    return x, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    return x.mean(dim=4).view(batch, channel, new_height, new_width)

def max_reduce(input, dim):
    return input.backend.max_reduce(input, dim)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        out = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, out = ctx.saved_values
        return (out == input) * grad_output, 0.0

        
def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, dim if isinstance(dim, Tensor) else tensor([dim]))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    input = input.to(dtype='float32')
    e = (input - Max.apply(input, tensor([dim]))).exp()
    partition = e.sum(dim=dim)
    return e / partition


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    e = input.to(dtype='float32')
    mx = Max.apply(e, tensor([dim]))
    lse = (e - mx).exp().sum(dim=dim).log() + mx
    return e - lse


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    return max(x, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input
    r = rand(input.shape, backend=input.backend)
    drop = rate < r
    return input * drop


def layer_norm(input: Tensor, eps: float = 1e-5) -> Tensor:
    # Calculate mean and variance along the last axis (features)
    batch, channel, height, width = input.shape
    
    mean = input.mean(dim=4).view(batch, channel, height, width)
    variance = input.var(dim=4).view(batch, channel, height, width)
    
    input_normalized = (input - mean) / (variance + eps)
    return input_normalized


def GELU(input: Tensor) -> Tensor: 
    """Applies the GELU activation function with 'tanh' approximation element-wise
    https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    """
    return 0.5 * input * (1 + (np.sqrt(2 / math.pi) * (input + 0.044715 * (input ** 3))).tanh())


def one_hot(input: Tensor, num_classes: int) -> Tensor:
    """Takes a Tensor containing indices of shape (*) and returns a tensor of shape (*, num_classes) 
    that contains zeros except a 1 where the index of last dimension matches the corresponding value of the input tensor.
    This is analogous to torch.nn.functional.one_hot (which contains helpful examples you may want to play around with)

    Hint: You may want to use a combination of np.eye, tensor_from_numpy, 
    """
    if input.backend.device_storage:
        return input.backend.ops.one_hot(input, num_classes)
    return tensor_from_numpy(
                np.eye(num_classes)[input.to_numpy().astype(int)], 
                backend=input.backend
            )


def logsumexp(input: Tensor, dim: int) -> Tensor:
    """Calculates logsumexp with logsumexp trick for numerical stability
    https://en.wikipedia.org/wiki/LogSumExp

    Args:
        input : The tensor to calculate logsumexp over
        dim : The dimension to reduce over

    Returns:
        out : The output tensor with the same number of dimensions as input (equiv. to keepdims=True)
            NOTE: minitorch functions/tensor functions typically keep dimensions if you provide a dimensions.
    """  
    ### BEGIN ASSIGN3_1
    e = input
    mx = max(e, dim)
    return (e - mx).exp().sum(dim=dim).log() + mx
    ### END ASSIGN3_1


def softmax_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Softmax + Cross Entropy Loss function with 'reduction=None'.
    Formula is provided in writeup.

    Args: 
        logits : (minibatch, C) Tensor of raw logits       
        target : (minibatch, ) Tensor of true labels 

    Returns: 
        loss : (minibatch, )
    """
    result = None
    batch_size = logits.shape[0]
    ### BEGIN ASSIGN3_1
    target_one_hot = one_hot(target, logits.shape[1])
    target_logits = (logits * target_one_hot).sum(dim=1)
    result = logsumexp(logits, dim=1) - target_logits
    ### END ASSIGN3_1
    return result.view(batch_size,)
