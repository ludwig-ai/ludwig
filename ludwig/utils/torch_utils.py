import math
import os
import warnings
from abc import abstractmethod
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Module, ModuleDict

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.strings_utils import SpecialSymbol

_TORCH_INIT_PARAMS: Optional[Tuple] = None


@DeveloperAPI
def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"

    if bool(os.environ.get("LUDWIG_ENABLE_MPS")):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            if not bool(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")):
                warnings.warn(
                    "LUDWIG_ENABLE_MPS is set and MPS is available, but PYTORCH_ENABLE_MPS_FALLBACK has not been set. "
                    "Depending on your model config, some operations may not be compatible. If errors occur, try "
                    "setting `PYTORCH_ENABLE_MPS_FALLBACK=1` and resubmitting."
                )
            return "mps"
        else:
            warnings.warn("LUDWIG_ENABLE_MPS is set but MPS is not available, falling back to CPU.")

    return "cpu"


DEVICE = get_torch_device()


@DeveloperAPI
def place_on_device(x, device):
    """Recursively places the input on the specified device."""
    if isinstance(x, list):
        return [place_on_device(xi, device) for xi in x]
    elif isinstance(x, dict):
        return {k: place_on_device(v, device) for k, v in x.items()}
    elif isinstance(x, set):
        return {place_on_device(xi, device) for xi in x}
    elif isinstance(x, tuple):
        return tuple(place_on_device(xi, device) for xi in x)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return x


@DeveloperAPI
def sequence_length_2D(sequence: torch.Tensor) -> torch.Tensor:
    """Returns the number of non-padding elements per sequence in batch.

    :param sequence: (torch.Tensor) A 2D tensor of shape [batch size x max sequence length].

    # Return
    :returns: (torch.Tensor) The count on non-zero elements per sequence.
    """
    used = (sequence != SpecialSymbol.PADDING.value).type(torch.int32)
    length = torch.sum(used, 1)
    return length


@DeveloperAPI
def sequence_length_3D(sequence: torch.Tensor) -> torch.Tensor:
    """Returns the number of non-zero elements per sequence in batch.

    :param sequence: (torch.Tensor) A 3D tensor of shape [batch size x max sequence length x hidden size].

    # Return
    :returns: (torch.Tensor) The count on non-zero elements per sequence.
    """
    used = torch.sign(torch.amax(torch.abs(sequence), dim=2))
    length = torch.sum(used, 1)
    length = length.int()
    return length


@DeveloperAPI
def sequence_mask(lengths: torch.Tensor, maxlen: Optional[int] = None, dtype: torch.dtype = torch.bool):
    """Returns a mask of shape (batch_size x maxlen), where mask[i] is True for each element up to lengths[i],
    otherwise False i.e. if maxlen=5 and lengths[i] = 3, mask[i] = [True, True True, False False].

    :param lengths: (torch.Tensor) A 1d integer tensor of shape [batch size].
    :param maxlen: (Optional[int]) The maximum sequence length.  If not specified, the max(lengths) is used.
    :param dtype: (type) The type to output.

    # Return
    :returns: (torch.Tensor) A sequence mask tensor of shape (batch_size x maxlen).
    """
    if maxlen is None:
        maxlen = lengths.max()
    matrix = torch.unsqueeze(lengths, dim=-1)
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    mask = row_vector < matrix
    mask = mask.type(dtype)
    return mask


@DeveloperAPI
def periodic(inputs: torch.Tensor, period: int) -> torch.Tensor:
    """Returns periodic representation assuming 0 is start of period."""
    return torch.cos(inputs * 2 * math.pi / period)


initializer_registry = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "constant": nn.init.constant_,
    "ones": nn.init.ones_,
    "zeros": nn.init.zeros_,
    "eye": nn.init.eye_,
    "dirac": nn.init.dirac_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "orthogonal": nn.init.orthogonal_,
    "sparse": nn.init.sparse_,
    "identity": nn.init.eye_,
}

activations = {
    "elu": nn.ELU,
    "leakyRelu": nn.LeakyReLU,
    "logSigmoid": nn.LogSigmoid,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    None: nn.Identity,
}


@DeveloperAPI
def get_activation(activation):
    return activations[activation]()


@DeveloperAPI
def reg_loss(model: nn.Module, regularizer: str, l1: float = 0.01, l2: float = 0.01):
    """Computes the regularization loss for a given model.

    Parameters:
        model: torch.nn.Module object to compute regularization loss for.
        regularizer: regularizer to use (currently l1, l2 and l1_l2 supported).
        l1: L1 regularization coefficient.
        l2: L2 regularization coefficient.

    Returns:
        Regularization loss for the model (float).
    """

    if regularizer == "l1":
        l1_reg = l1 * sum(torch.abs(p).sum() for p in model.parameters())
        return l1_reg
    if regularizer == "l2":
        l2_reg = l2 * sum(torch.square(p).sum() for p in model.parameters())
        return l2_reg
    if regularizer == "l1_l2":
        l1_reg = l1 * sum(torch.abs(p).sum() for p in model.parameters())
        l2_reg = l2 * sum(torch.square(p).sum() for p in model.parameters())
        return l1_reg + l2_reg


@DeveloperAPI
class LudwigModule(Module):
    def __init__(self):
        super().__init__()
        self._losses = {}
        self.register_buffer("device_tensor", torch.zeros(0))

    @property
    def device(self):
        return self.device_tensor.device

    def losses(self):
        collected_losses = []
        for loss in self._losses.values():
            collected_losses.append(loss)

        for child in self.children():
            if isinstance(child, LudwigModule):
                collected_losses.extend(child.losses())
            elif isinstance(child, ModuleDict):
                for c in child.values():
                    if hasattr(c, "losses"):  # Some modules, i.e. SequenceReducers, don't have losses.
                        collected_losses.extend(c.losses())
            elif isinstance(child, Module):
                pass
            else:
                raise ValueError

        return collected_losses

    def update_loss(self, key: str, loss: torch.Tensor):
        """This should be called in the forward pass to add a custom loss term to the combined loss."""
        self._losses[key] = loss

    @property
    def input_dtype(self):
        return torch.float32

    @property
    @abstractmethod
    def input_shape(self) -> torch.Size:
        """Returns size of the input tensor without the batch dimension."""
        pass
        # raise NotImplementedError("Abstract class.")

    @property
    def output_shape(self) -> torch.Size:
        """Returns size of the output tensor without the batch dimension."""
        return self._computed_output_shape()

    @lru_cache(maxsize=1)
    def _computed_output_shape(self) -> torch.Size:
        dummy_input = torch.rand(2, *self.input_shape, device=self.device)
        output_tensor = self.forward(dummy_input.type(self.input_dtype))

        if isinstance(output_tensor, torch.Tensor):
            return output_tensor.size()[1:]
        elif isinstance(output_tensor, dict) and "encoder_output" in output_tensor:
            return output_tensor["encoder_output"].size()[1:]
        else:
            raise ValueError("Unknown output tensor type.")


def freeze_parameters(module: nn.Module):
    """Freezes the parameters of a torch module."""
    for p in module.parameters():
        p.requires_grad = False


@DeveloperAPI
class FreezeModule(nn.Module):
    def __init__(self, module: nn.Module, frozen: bool):
        super().__init__()
        if frozen:
            freeze_parameters(module)
            module.eval()
        else:
            module.train()
        self.module = module
        self.frozen = frozen

    def train(self, mode: bool = True):
        if self.frozen:
            # Ignores any attempt to set params trainable
            return self

        return super().train(mode)


@DeveloperAPI
class Dense(LudwigModule):
    def __init__(
        self,
        input_size,
        output_size,
        use_bias=True,
        weights_initializer="xavier_uniform",
        bias_initializer="zeros",
    ):
        super().__init__()
        self.dense = nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias)
        weights_initializer = initializer_registry[weights_initializer]
        weights_initializer(self.dense.weight)

        if use_bias:
            bias_initializer = initializer_registry[bias_initializer]
            bias_initializer(self.dense.bias)

    @property
    def input_shape(self) -> torch.Size:
        return self.dense.input_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.squeeze(self.dense(input), dim=-1)
        return output


@DeveloperAPI
def initialize_pytorch(
    gpus: Optional[Union[int, str, List[int]]] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    local_rank: int = 0,
    local_size: int = 1,
):
    param_tuple = (gpus, gpu_memory_limit, allow_parallel_threads, local_rank, local_size)
    if _TORCH_INIT_PARAMS is not None:
        if _TORCH_INIT_PARAMS != param_tuple:
            warnings.warn(
                "PyTorch has already been initialized. Changes to `gpus`, "
                "`gpu_memory_limit`, and `allow_parallel_threads` will be ignored. "
                "Start a new Python process to modify these values."
            )
        return

    # For reproducivility / determinism, set parallel threads to 1.
    # For performance, leave unset to allow PyTorch to select the best value automatically.
    if not allow_parallel_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    gpu_device_count = torch.cuda.device_count()
    if local_size > 1 and gpus is None:
        if 0 < gpu_device_count < local_size:
            warnings.warn(
                f"Distributed: disabling GPU support! This host is running with "
                f"{local_size} worker processes but only {gpu_device_count} "
                f"GPUs. To enable GPU training, reduce the number of worker processes "
                f"on this host to match the number of GPUs."
            )
            gpus = [-1]
        else:
            gpus = [local_rank]

    if isinstance(gpus, int):
        gpus = [gpus]
    elif isinstance(gpus, str):
        gpus = gpus.strip()
        gpus = [int(g) for g in gpus.split(",")]

    if gpus and len(gpus) == 1 and gpus[0] == -1:
        # CUDA_VISIBLE_DEVICES syntax for disabling all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif torch.cuda.is_available():
        # Set visible devices so GPU utilization is isolated
        # (no GPU contention between workers).
        if gpus is not None:
            if len(gpus) == 1:
                torch.cuda.set_device(gpus[0])
            elif len(gpus) > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)

        # Limit the amount of memory that can be consumed per GPU
        if gpu_memory_limit is not None:
            for gpu in gpus or range(gpu_device_count):
                torch.cuda.memory.set_per_process_memory_fraction(gpu_memory_limit, gpu)

    _set_torch_init_params(param_tuple)


def _set_torch_init_params(params: Optional[Tuple]):
    global _TORCH_INIT_PARAMS
    _TORCH_INIT_PARAMS = params


def _get_torch_init_params() -> Optional[Tuple]:
    return _TORCH_INIT_PARAMS


@DeveloperAPI
def model_size(model: nn.Module):
    """Computes PyTorch model size in bytes."""
    size = 0
    size += sum(param.nelement() * param.element_size() for param in model.parameters())
    size += sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    return size
