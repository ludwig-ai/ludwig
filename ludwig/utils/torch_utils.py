from abc import abstractmethod
from functools import lru_cache

import math
import torch
from torch import nn
from torch.nn import Module, ModuleDict


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


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
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


def reg_loss(input_tensor, regularizer, l1=0.01, l2=0.01):
    l1_reg = l1 * torch.sum(torch.abs(input_tensor))
    l2_reg = l2 * torch.sum(torch.square(input_tensor))

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
                print(type(child))
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
        """ Returns size of the input tensor without the batch dimension."""
        raise NotImplementedError('Abstract class.')

    @property
    def output_shape(self) -> torch.Size:
        """ Returns size of the output tensor without the batch dimension."""
        return self._compute_output_shape()

    @lru_cache(maxsize=1)
    def _compute_output_shape(self) -> torch.Size:
        output_tensor = self.forward(
            torch.rand(2, *self.input_shape).type(self.input_dtype))
        return output_tensor.size()[1:]


class Dense(LudwigModule):
    def __init__(
        self,
        input_size,
        output_size,
        use_bias=True,
        weights_initializer='xavier_uniform',
        bias_initializer='zeros',
        weights_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
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

        if weights_regularizer:
            self.add_loss(lambda: reg_loss(
                self.dense.weight, weights_regularizer))

        if bias_regularizer:
            self.add_loss(lambda: reg_loss(self.dense.bias, bias_regularizer))

        if activity_regularizer:
            # Handle in forward call
            self.add_loss(lambda: self.activation_loss)

        self.activity_regularizer = activity_regularizer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[0]
        output = torch.squeeze(self.dense(input), dim=-1)
        if self.activity_regularizer:
            self.activation_loss = reg_loss(
                output, self.activity_regularizer) / batch_size
        return output
