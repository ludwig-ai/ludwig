import os
import warnings
from abc import abstractmethod
from functools import lru_cache

from typing import Optional, List, Tuple, Union

import math
import torch
from torch import nn
from torch.nn import Module, ModuleDict
from torch.autograd import Function


_TORCH_INIT_PARAMS: Optional[Tuple] = None


def sequence_length_2D(sequence: torch.Tensor) -> torch.Tensor:
    """ Returns the number of non-zero elements per sequence. """
    used = (sequence != 0).type(torch.int32)
    length = torch.sum(used, 1)
    return length


def sequence_length_3D(sequence: torch.Tensor) -> torch.Tensor:
    used = torch.sign(torch.amax(torch.abs(sequence), dim=2))
    length = torch.sum(used, 1)
    length = length.int()
    return length


def sequence_mask(
        lengths: torch.Tensor,
        maxlen: Optional[int] = None, dtype: torch.dtype = torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


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
    'identity': nn.init.eye_,
    None: nn.init.xavier_uniform_,

}

activations = {
    "elu": nn.ELU,
    "leakyRelu": nn.LeakyReLU,
    "logSigmoid": nn.LogSigmoid,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    None: nn.Identity
}


def get_activation(activation):
    return activations[activation]()


def reg_loss(
        model: nn.Module,
        regularizer: str,
        l1: float = 0.01,
        l2: float = 0.01
):
    """
    Computes the regularization loss for a given model.

    Parameters:
        model: torch.nn.Module object to compute regularization loss for.
        regularizer: regularizer to use (currently l1, l2 and l1_l2 supported).
        l1: L1 regularization coefficient.
        l2: L2 regularization coefficient.

    Returns:
        Regularization loss for the model (float).
    """

    l1_reg = l1 * sum(torch.abs(p).sum() for p in model.parameters())
    l2_reg = l2 * sum(torch.square(p).sum() for p in model.parameters())

    if regularizer == "l1":
        return l1_reg
    if regularizer == "l2":
        return l2_reg
    if regularizer == "l1_l2":
        return l1_reg + l2_reg


class LudwigModule(Module):
    def __init__(self):
        super().__init__()
        self._callable_losses = []
        self.register_buffer('device_tensor', torch.zeros(0))

    @property
    def device(self):
        return self.device_tensor.device

    def losses(self):
        collected_losses = []
        for loss_fn in self._callable_losses:
            collected_losses.append(loss_fn())

        for child in self.children():
            if isinstance(child, LudwigModule):
                collected_losses.extend(child.losses())
            elif isinstance(child, ModuleDict):
                for c in child.values():
                    collected_losses.extend(c.losses())
            elif isinstance(child, Module):
                pass
            else:
                raise ValueError

        return collected_losses

    def add_loss(self, loss):
        if callable(loss):
            self._callable_losses.append(loss)

    @property
    def input_dtype(self):
        return torch.float32

    @property
    @abstractmethod
    def input_shape(self) -> torch.Size:
        """Returns size of the input tensor without the batch dimension."""
        raise NotImplementedError('Abstract class.')

    @property
    def output_shape(self) -> torch.Size:
        """Returns size of the output tensor without the batch dimension."""
        return self._compute_output_shape()

    @lru_cache(maxsize=1)
    def _compute_output_shape(self) -> torch.Size:

        dummy_input = torch.rand(2, *self.input_shape, device=self.device)

        output_tensor = self.forward(dummy_input.type(self.input_dtype))
        if isinstance(output_tensor, torch.Tensor):
            return output_tensor.size()[1:]
        elif isinstance(output_tensor, dict) and 'encoder_output' in output_tensor:
            return output_tensor['encoder_output'].size()[1:]
        else:
            raise ValueError('Unknown output tensor type.')


class Dense(LudwigModule):
    def __init__(
        self,
        input_size,
        output_size,
        use_bias=True,
        weights_initializer='xavier_uniform',
        bias_initializer='zeros',
    ):
        super().__init__()
        self.dense = nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=use_bias
        )
        weights_initializer = initializer_registry[weights_initializer]
        weights_initializer(self.dense.weight)

        bias_initializer = initializer_registry[bias_initializer]
        bias_initializer(self.dense.bias)

    @property
    def input_shape(self) -> torch.Size:
        return self.dense.input_shape

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        output = torch.squeeze(self.dense(input), dim=-1)
        return output


# sparsemax implementation: https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/sparsemax.py
# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax
        Returns
        -------
        output : torch.Tensor
            same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = SparsemaxFunction._threshold_and_support(input,
                                                                  dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax
        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor
        """

        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum

        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(torch.nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


def initialize_pytorch(
        gpus: Optional[Union[int, str, List[int]]] = None,
        gpu_memory_limit: Optional[float] = None,
        allow_parallel_threads: bool = True,
        horovod: Optional["horovod.torch"] = None
):
    use_horovod = horovod is not None
    param_tuple = (gpus, gpu_memory_limit, allow_parallel_threads, use_horovod)
    if _TORCH_INIT_PARAMS is not None:
        if _TORCH_INIT_PARAMS != param_tuple:
            warnings.warn(
                'PyTorch has already been initialized. Changes to `gpus`, '
                '`gpu_memory_limit`, and `allow_parallel_threads` will be ignored. '
                'Start a new Python process to modify these values.')
        return

    # For reproducivility / determinism, set parallel threads to 1.
    # For performance, leave unset to allow PyTorch to select the best value automatically.
    if not allow_parallel_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    gpu_device_count = torch.cuda.device_count()
    if horovod is not None and gpus is None:
        if 0 < gpu_device_count < horovod.local_size():
            warnings.warn(
                f'Horovod: disabling GPU support! This host is running with '
                f'{horovod.local_size()} worker processes but only {gpu_device_count} '
                f'GPUs. To enable GPU training, reduce the number of worker processes '
                f'on this host to match the number of GPUs.')
            gpus = [-1]
        else:
            gpus = [horovod.local_rank()]

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
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    str(i) for i in gpus)

        # Limit the amount of memory that can be consumed per GPU
        if gpu_memory_limit is not None:
            for gpu in gpus or range(gpu_device_count):
                torch.cuda.memory.set_per_process_memory_fraction(
                    gpu_memory_limit, gpu
                )

    _set_torch_init_params(param_tuple)


def _set_torch_init_params(params: Optional[Tuple]):
    global _TORCH_INIT_PARAMS
    _TORCH_INIT_PARAMS = params


def _get_torch_init_params() -> Optional[Tuple]:
    return _TORCH_INIT_PARAMS
