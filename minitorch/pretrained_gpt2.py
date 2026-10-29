"""GPT-2-family inference adapter. All timed computation runs in MiniTorch.

Transformers is used only for checkpoint/tokenizer loading and an untimed CPU
reference. GPT-2 Conv1D matrices already use MiniTorch's [in,out] orientation.
"""
import numpy as np
from .module import Module, Parameter
from .tensor_functions import tensor_from_numpy
from .modules_basic import Linear, Dropout, LayerNorm1d
from .modules_transformer import DecoderLM, MultiHeadAttention, TransformerLayer, FeedForward
from .autodiff import is_grad_enabled, no_grad
from .cuda_resident_ops import CudaResidentOps as Ops, pointer, call, P, I
from .kv_cache import _library


class GatherEmbedding(Module):
    def __init__(self, weight, backend):
        super().__init__()
        self.backend=backend
        self.weights=Parameter(tensor_from_numpy(weight,backend))
        self.num_embeddings,self.embedding_dim=weight.shape

    def forward(self, ids):
        if is_grad_enabled():
            raise ValueError('Pretrained embedding adapter is inference-only; use no_grad')
        if ids.backend is not self.backend or ids.dtype!='float32':
            raise ValueError('Embedding IDs must be FP32 on the model backend')
        out=Ops.empty((*ids.shape,self.embedding_dim),self.backend)
        ids=ids.contiguous()
        call(_library(),'mt_embedding_gather',[P,P,P,I,I,I],pointer(out),
            pointer(self.weights.value),pointer(ids),ids.size,self.num_embeddings,self.embedding_dim)
        return out


class TiedHead(Module):
    def __init__(self, embedding):
        super().__init__()
        # Share the logical parameter; materialize the transposed GEMM layout once.
        self.weights=embedding.weights
        self._projection=None
        self._version=None

    def forward(self, x):
        w=self.weights.value
        version=(id(w),w._tensor._storage.owner.version)
        if self._version!=version:
            with no_grad(): self._projection=w.permute(1,0).contiguous()
            self._version=version
        return x @ self._projection


def from_gpt2_state(config, state, backend):
    """Map a validated standard GPT-2 checkpoint, without random initialization."""
    if not backend.device_storage:
        raise ValueError('Pretrained adapter requires resident device storage')
    if (config.model_type!='gpt2' or config.activation_function!='gelu_new'
        or not getattr(config,'scale_attn_weights',True)
        or getattr(config,'scale_attn_by_inverse_layer_idx',False)
        or getattr(config,'reorder_and_upcast_attn',False)
        or getattr(config,'add_cross_attention',False)
        or not getattr(config,'tie_word_embeddings',True)):
        raise ValueError('Supported architecture: standard GPT-2 with gelu_new and tied embeddings')
    h,heads=config.n_embd,config.n_head
    if h<1 or heads<1 or h%heads or h//heads>128 or config.n_layer<1:
        raise ValueError('GPT-2 head_dim must be <=128; positive layer count required')
    inner=config.n_inner or 4*h
    def array(name,shape):
        if name not in state:
            raise ValueError(f'Missing checkpoint parameter: {name}')
        a=state[name]
        if hasattr(a,'detach'): a=a.detach().cpu().float().numpy()
        a=np.asarray(a,dtype=np.float32)
        if a.shape!=shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid checkpoint parameter: {name}, expected {shape}')
        return np.ascontiguousarray(a)
    # Validate complete mapping before allocating device weights.
    shapes={'transformer.wte.weight':(config.vocab_size,h),
            'transformer.wpe.weight':(config.n_positions,h),
            'transformer.ln_f.weight':(h,),'transformer.ln_f.bias':(h,)}
    for i in range(config.n_layer):
        prefix=f'transformer.h.{i}.'
        for ln in ('ln_1','ln_2'):
            shapes[prefix+ln+'.weight']=(h,);shapes[prefix+ln+'.bias']=(h,)
        for name,shape in [('attn.c_attn.weight',(h,3*h)),('attn.c_attn.bias',(3*h,)),
            ('attn.c_proj.weight',(h,h)),('attn.c_proj.bias',(h,)),
            ('mlp.c_fc.weight',(h,inner)),('mlp.c_fc.bias',(inner,)),
            ('mlp.c_proj.weight',(inner,h)),('mlp.c_proj.bias',(h,))]: shapes[prefix+name]=shape
    arrays={name:array(name,shape) for name,shape in shapes.items()}
    if 'lm_head.weight' in state and not np.array_equal(array('lm_head.weight',(config.vocab_size,h)),arrays['transformer.wte.weight']):
        raise ValueError('Expected tied GPT-2 output embeddings')
    def linear(weight,bias):
        obj=Linear.__new__(Linear);Module.__init__(obj)
        obj.backend=backend;obj.in_size,obj.out_size=weight.shape
        obj.weights=Parameter(tensor_from_numpy(weight,backend))
        obj.bias=Parameter(tensor_from_numpy(bias,backend))
        return obj
    def norm(prefix):
        obj=LayerNorm1d(h,eps=config.layer_norm_epsilon,backend=backend)
        obj.weights.update(tensor_from_numpy(arrays[prefix+'.weight'],backend))
        obj.bias.update(tensor_from_numpy(arrays[prefix+'.bias'],backend))
        return obj
    model=PretrainedGPT2.__new__(PretrainedGPT2);Module.__init__(model)
    model.backend=backend;model.n_embd=h;model.n_vocab=config.vocab_size
    model.n_positions=config.n_positions;model.layer_count=config.n_layer
    model.token_embeddings=GatherEmbedding(arrays['transformer.wte.weight'],backend)
    model.position_embeddings=GatherEmbedding(arrays['transformer.wpe.weight'],backend)
    model.dropout=Dropout(0)
    model.ln=norm('transformer.ln_f');model.lm_head=TiedHead(model.token_embeddings)
    for i in range(config.n_layer):
        p=f'transformer.h.{i}.'
        block=TransformerLayer.__new__(TransformerLayer);Module.__init__(block)
        block.n_embd=h;block.ln_1=norm(p+'ln_1');block.ln_2=norm(p+'ln_2')
        attn=MultiHeadAttention.__new__(MultiHeadAttention);Module.__init__(attn)
        attn.backend=backend;attn.n_embd=h;attn.n_head=heads;attn.attn_hidden_dim=h//heads
        attn.causal=True;attn.attention_impl='flash';attn.dropout=Dropout(0)
        w,b=arrays[p+'attn.c_attn.weight'],arrays[p+'attn.c_attn.bias']
        for j,name in enumerate(('q_projection','k_projection','v_projection')):
            setattr(attn,name,linear(w[:,j*h:(j+1)*h].copy(),b[j*h:(j+1)*h].copy()))
        attn.out_projection=linear(arrays[p+'attn.c_proj.weight'],arrays[p+'attn.c_proj.bias'])
        block.attention=attn
        ff=FeedForward.__new__(FeedForward);Module.__init__(ff)
        ff.linear_in=linear(arrays[p+'mlp.c_fc.weight'],arrays[p+'mlp.c_fc.bias'])
        ff.linear_out=linear(arrays[p+'mlp.c_proj.weight'],arrays[p+'mlp.c_proj.bias'])
        ff.dropout=Dropout(0);block.ff=ff
        setattr(model,f'block_{i}',block)
    model.eval()
    return model


class PretrainedGPT2(DecoderLM):
    def transformer_layers(self):
        return tuple(getattr(self,f'block_{i}') for i in range(self.layer_count))

    def forward(self, idx, last_only=False):
        if self.training or is_grad_enabled():
            raise ValueError('PretrainedGPT2 is an inference adapter; use eval and no_grad')
        return super().forward(idx,last_only=last_only)


def load_pretrained(model_name='distilbert/distilgpt2',revision='main',cache_dir=None,local_files_only=False):
    """Load only audited built-in GPT2 architecture and safetensors weights on CPU."""
    from transformers import AutoConfig,AutoTokenizer,GPT2LMHeadModel
    options=dict(revision=revision,cache_dir=cache_dir,local_files_only=local_files_only,trust_remote_code=False)
    config=AutoConfig.from_pretrained(model_name,**options)
    if config.model_type!='gpt2':
        raise ValueError('Only GPT-2 family checkpoints are supported (e.g. distilbert/distilgpt2)')
    model=GPT2LMHeadModel.from_pretrained(model_name,config=config,use_safetensors=True,
        torch_dtype='float32',attn_implementation='eager',**options).cpu().eval()
    tokenizer=AutoTokenizer.from_pretrained(model_name,**options)
    return model,tokenizer
