"""Explicit, per-model execution choices (no method monkey-patching)."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KernelConfig:
    attention_impl: str = "fused_softmax"
    layernorm_impl: str = "fused"
    storage_mode: str = "host"
    memory_mode: str = "default"
    dtype: str = "float32"
    use_kv_cache: bool = False
    pool_cache_mb: int = 1024
    optimizer_impl: str = 'adam'

    def __post_init__(self):
        supported = {
            "attention_impl": ("reference", "fused_softmax", "flash"),
            "layernorm_impl": ("reference", "fused"),
            "storage_mode": ("host", "device"),
            "memory_mode": ("default", "pooled", "planned"),
            "dtype": ("float32",),
            "optimizer_impl": ('adam', 'fused_adam'),
        }
        for name, choices in supported.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name}={getattr(self, name)!r} is not implemented; choose {choices}")
        if type(self.use_kv_cache) is not bool:
            raise ValueError('use_kv_cache must be a bool')
        if self.use_kv_cache and (self.storage_mode != 'device' or self.attention_impl != 'flash'):
            raise ValueError('KV cache requires storage_mode=device and attention_impl=flash')
        if self.attention_impl == "flash" and self.storage_mode != "device":
            raise ValueError("attention_impl=flash requires storage_mode=device")
        if self.memory_mode != 'default' and self.storage_mode != 'device':
            raise ValueError('Memory pooling requires storage_mode=device')
        if self.optimizer_impl == 'fused_adam' and self.storage_mode != 'device':
            raise ValueError('optimizer_impl=fused_adam requires storage_mode=device')
        if isinstance(self.pool_cache_mb, bool) or not isinstance(self.pool_cache_mb, int) or self.pool_cache_mb < 0:
            raise ValueError('pool_cache_mb must be a nonnegative integer')

    @classmethod
    def resolve(cls, use_fused_kernel=True, attention_impl=None, layernorm_impl=None, **kwargs):
        if not isinstance(use_fused_kernel, bool):
            raise ValueError("use_fused_kernel must be True or False")
        return cls(
            attention_impl=attention_impl or ("fused_softmax" if use_fused_kernel else "reference"),
            layernorm_impl=layernorm_impl or ("fused" if use_fused_kernel else "reference"),
            **kwargs,
        )

    def to_dict(self):
        return asdict(self)


def configure_kernels(model, config):
    from .modules_basic import LayerNorm1d
    from .modules_transformer import MultiHeadAttention
    if isinstance(config, bool):
        config = KernelConfig.resolve(config)
    if not isinstance(config, KernelConfig):
        raise TypeError("config must be KernelConfig or bool")
    if isinstance(model, MultiHeadAttention):
        model.attention_impl = config.attention_impl
    if isinstance(model, LayerNorm1d):
        model.layernorm_impl = config.layernorm_impl
    for child in model.modules():
        configure_kernels(child, config)


def make_backend(config):
    from .tensor_ops import TensorBackend
    if config.storage_mode == "device":
        from .cuda_resident_ops import CudaResidentOps
        backend = TensorBackend(CudaResidentOps)
        backend.memory_mode = config.memory_mode
        if config.memory_mode != 'default':
            from .memory_pool import MemoryPool
            from .cuda_storage import runtime
            backend.memory_pool = MemoryPool(runtime(), config.pool_cache_mb*1024**2)
            if config.memory_mode == 'planned':
                from .workspace import Workspace
                backend.workspace = Workspace(backend.memory_pool)
        return backend
    from .cuda_kernel_ops import CudaKernelOps
    return TensorBackend(CudaKernelOps)


def make_optimizer(parameters, config, **kwargs):
    from .optim import Adam
    from .packed_adam import FusedAdam
    cls = FusedAdam if config.optimizer_impl == 'fused_adam' else Adam
    return cls(parameters, **kwargs)


def attention_dispatch_counts(model):
    """Count the actually selected paths from the most recent model forward."""
    from .modules_transformer import MultiHeadAttention
    counts=dict(reference=0,fused_softmax=0,flash=0)
    def visit(module):
        if isinstance(module,MultiHeadAttention):
            selected=module.__dict__.get('last_attention_impl')
            if selected in counts: counts[selected]+=1
        for child in module.modules(): visit(child)
    visit(model)
    return counts
