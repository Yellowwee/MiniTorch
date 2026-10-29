import numpy as np
import warnings
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
    flash_attention,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attention_impl = "fused_softmax"
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_head * self.attn_hidden_dim, bias=bias, backend=self.backend)
        self.k_projection = Linear(n_embd, n_head * self.attn_hidden_dim, bias=bias, backend=self.backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.dropout = Dropout(p_dropout)

        assert n_embd % n_head == 0
        ### END ASSIGN3_3

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        if self.backend.device_storage:
            return self.backend.ops.sequence((1,1,seq_len,seq_len),self.backend,width=seq_len)
        mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x, transpose_key=True):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        x_2d = x.contiguous().view(batch_size * seq_len, n_embd)
        q = self.q_projection(x_2d)
        k = self.k_projection(x_2d)
        v = self.v_projection(x_2d)

        q = (q.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous())
        k = (k.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous())
        v = (v.contiguous().view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3).contiguous())

        kT = k.permute(0, 1, 3, 2).contiguous() if transpose_key else k
        ### END ASSIGN3_3
        return q, kT, v
    
    def _flash_supported(self):
        return (self.backend.device_storage and self.attn_hidden_dim <= 128
                and (not self.training or self.dropout.p_dropout == 0))

    def self_attention(self, q, kT, v, key_transposed=True):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        k_dim = kT.shape[-2] if key_transposed else kT.shape[-1]
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        if self.attention_impl == 'flash' and self._flash_supported():
            self.last_attention_impl = 'flash'
            k = kT.permute(0,1,3,2) if key_transposed else kT
            result = flash_attention(q,k,v,causal=self.causal)
        else:
            if self.attention_impl == 'flash':
                warnings.warn('FlashAttention requires device storage, head_dim<=128 and attention dropout=0; using the existing attention path with dropout preserved.', RuntimeWarning, stacklevel=2)
            kt = kT if key_transposed else kT.permute(0,1,3,2)
            logits = (q @ kt) / np.sqrt(q_dim)
            if self.causal:
                logits += self.create_causal_mask(queries_len)
            if self.attention_impl in ('fused_softmax','flash') and logits.shape[-1] <= 1024:
                self.last_attention_impl = 'fused_softmax'
                padding_mask = logits.zeros((batch_size, logits.shape[-1]))
                attention = logits.contiguous().attn_softmax(padding_mask)
            else:
                self.last_attention_impl = 'reference'
                attention = softmax(logits, dim=3)
            result = self.dropout(attention) @ v
        result = (result.permute(0, 2, 1, 3).contiguous().view(batch_size * queries_len, self.n_embd))
        result = self.out_projection(result)
        ### END ASSIGN3_3
        return result.contiguous().view(batch_size, queries_len, self.n_embd)

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        transpose_key = not (self.attention_impl == 'flash' and self._flash_supported())
        q, kT, v = self.project_to_query_key_value(x, transpose_key=transpose_key)
        result = self.self_attention(q, kT, v, key_transposed=transpose_key)
        return result
        ### END ASSIGN3_3

    def forward_cached(self, x, cache, layer_index, phase=None):
        q,k,v = self.project_to_query_key_value(x, transpose_key=False)
        if phase not in (None,'prefill','decode'): raise ValueError('Unknown inference phase')
        entry=cache.attention if phase is None else cache.prefill_attention if phase=='prefill' else cache.decode_attention
        result = entry(layer_index,q,k,v)
        self.last_inference_path=cache.last_attention_path
        self.last_attention_impl = 'flash'
        b,_,n,_ = q.shape
        result = result.permute(0,2,1,3).contiguous().view(b*n,self.n_embd)
        return self.out_projection(result).view(b,n,self.n_embd)


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.n_embd = n_embd
        self.ln_1 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

    def forward(self, x, cache=None, layer_index=None, phase=None):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        assert n_embd == self.n_embd
        # Preserve FP32 residual accumulation under mixed-precision attention.
        x = x.to(dtype='float32')
        normalized = (x.contiguous().view(batch_size * seq_len, n_embd))
        normalized = self.ln_1(normalized)
        normalized = normalized.contiguous().view(batch_size, seq_len, n_embd)
        
        x = x + (self.attention(normalized) if cache is None else
                 self.attention.forward_cached(normalized,cache,layer_index,phase))
        normalized = (x.contiguous().view(batch_size * seq_len, n_embd))
        normalized = self.ln_2(normalized)
        normalized = normalized.contiguous().view(batch_size, seq_len, n_embd)

        x = x + self.ff(normalized)
        return x
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        self.n_positions = n_positions
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend)
        self.lm_head = Linear(n_embd, n_vocab, bias, backend)
        ### END ASSIGN3_3
    
    def transformer_layers(self):
        return (self.t_layer_1,self.t_layer_2,self.t_layer_3,self.t_layer_4)

    def create_kv_cache(self, batch_size=1, capacity=None, *, cache_impl='contiguous',
                        page_size=16, page_pool=None, kv_budget_bytes=256*1024**2, pages_per_slab=4,paged_prefill_impl='flash'):
        if cache_impl == 'paged':
            from .paged_kv_cache import PagedKVCache
            return PagedKVCache(self,batch_size,capacity,page_size=page_size,page_pool=page_pool,
                                kv_budget_bytes=kv_budget_bytes,pages_per_slab=pages_per_slab,paged_prefill_impl=paged_prefill_impl)
        if cache_impl != 'contiguous' or page_pool is not None:
            raise ValueError('Choose cache_impl=contiguous or paged; page_pool requires paged')
        from .kv_cache import KVCache
        return KVCache(self,batch_size,capacity)

    def prefill(self, idx, cache=None):
        """Evaluate a prompt, returning last-position logits and a populated cache."""
        from .autodiff import no_grad
        with no_grad():
            owns_cache = cache is None
            cache = self.create_kv_cache(idx.shape[0]) if owns_cache else cache
            try:
                return self._cached_forward(idx,cache,True),cache
            except Exception:
                if owns_cache: cache.close()
                raise

    def decode_step(self, idx, cache):
        """Process exactly one new token per batch; logits have shape [B,1,V]."""
        from .autodiff import no_grad
        with no_grad():
            return self._cached_forward(idx,cache,False)

    def decode_batch(self, idx, caches):
        """Decode one token per independent paged request in a single model batch.

        Rows may have different context lengths; all caches share one page pool.
        No cache compaction or KV gather is performed when batch membership changes.
        """
        from .autodiff import no_grad
        from .paged_decode import PagedDecodeBatch
        with no_grad():
            with PagedDecodeBatch(self,idx,caches) as batch:
                batch.prepare()
                x = self.token_embeddings(idx) + self.position_embeddings(batch.position_ids)
                x = self.dropout(x)
                for i,layer in enumerate(self.transformer_layers()):
                    x = layer(x,batch,i,phase='decode')
                logits = self.lm_head(self.ln(x.view(batch.batch_size,self.n_embd))).view(batch.batch_size,1,self.n_vocab)
                batch.commit()
                return logits

    def _cached_forward(self, idx, cache, prefill):
        from .kv_cache import last_position
        cache.validate(self,idx,prefill)
        b,n = idx.shape
        offset = cache.length
        try:
            cache.prepare_append(n)
            ids = self.backend.ops.sequence((1,n),self.backend) + offset
            x = self.token_embeddings(idx) + self.position_embeddings(ids)
            x = self.dropout(x)
            for i,layer in enumerate(self.transformer_layers()):
                x = layer(x,cache,i,phase='prefill' if prefill else 'decode')
            x = last_position(x).view(b,self.n_embd)
            logits = self.lm_head(self.ln(x)).view(b,1,self.n_vocab)
            cache.commit_append()
            return logits
        except Exception:
            cache.abort_append()  # Roll back page reservations; reset before retry.
            raise

    def forward(self, idx, last_only=False):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        if not 0 < seq_len <= self.n_positions:
            raise ValueError('Input sequence exceeds position embedding capacity or is empty')
        if last_only:
            from .autodiff import is_grad_enabled
            if is_grad_enabled():
                raise ValueError('last_only requires no_grad')

        ### BEGIN ASSIGN3_3
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        #    - Pass through positional embedding layer
        #    - Ensure output shape is (1, seq_len, n_embd)
        # 3. Add token and positional embeddings
        # 4. Apply dropout
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        # 6. Apply final layer normalization
        # 7. Project to vocabulary size using lm_head
        token_embeddings = self.token_embeddings(idx)
        if self.backend.device_storage:
            ids = self.backend.ops.sequence((1,seq_len),self.backend)
        else:
            ids = tensor([list(range(seq_len))], backend=self.backend)
        position_embeddings = self.position_embeddings(ids)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        for layer in self.transformer_layers():
            x = layer(x)
        if last_only:
            if self.backend.device_storage:
                from .kv_cache import last_position
                x = last_position(x)
            else:
                x = tensor_from_numpy(x.to_numpy()[:,-1:,:].copy(),backend=self.backend)
            seq_len = 1
        x = (x.contiguous().view(batch_size * seq_len, self.n_embd))
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits.contiguous().view(batch_size, seq_len, self.n_vocab)
        ### END ASSIGN3_3
