# MODULE 0
# MODULE 1
import minitorch.scalar_functions as scalar_functions  # noqa: F401,F403

from .autodiff import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
# from .cuda_kernel_ops import *
from .datasets import datasets  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403

# MODULE 3
from .fast_ops import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403

# MODULE 4
from .nn import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .scalar import Scalar, ScalarHistory, derivative_check  # noqa: F401,F403
from .scalar_functions import ScalarFunction  # noqa: F401,F403

# MODULE 2
from .tensor import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403

# ASSIGNMENT 2
from .modules_transformer import *
from .cuda_resident_ops import CudaResidentOps
from .packed_adam import FusedAdam
from .amp import autocast, GradScaler
from .autodiff import no_grad, set_grad_enabled, is_grad_enabled
from .kernel_config import KernelConfig, configure_kernels

version = "0.4"


def __getattr__(name):
    # Preserve the public legacy backend without loading CUDA for CPU users.
    if name == "CudaKernelOps":
        from .cuda_kernel_ops import CudaKernelOps
        return CudaKernelOps
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
