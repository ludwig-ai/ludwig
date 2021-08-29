from torch.nn import Module, ModuleDict
import torch
from torch.nn.init import (uniform_, normal_, constant_, ones_,  zeros_, eye_, dirac_,
        xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_)
from torch.nn import (ELU, LeakyReLU, LogSigmoid, ReLU, Sigmoid, Tanh, Softmax, Linear)

def sequence_length_3D(sequence):
    used = torch.sign(torch.max(torch.abs(sequence), dim=2))
    length = torch.sum(used, 1)
    length = length.type(torch.int32)
    return length

def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


initializers = {
        "uniform": uniform_,
        "normal": normal_,
        "constant": constant_,
        "ones": ones_,
        "zeros": zeros_,
        "eye": eye_,
        "dirac": dirac_,
        "xavier_uniform": xavier_uniform_,
        "xavier_normal": xavier_normal_,
        "kaiming_uniform": kaiming_uniform_,
        "kaiming_normal": kaiming_normal_,
        "orthogonal": orthogonal_,
        "sparse": sparse_
}

activations = {
        "elu": ELU,
        "leakyRelu": LeakyReLU,
        "logSigmoid": LogSigmoid,
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax
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


class LudwigLayer(Module):
    def __init__(self):
        super(LudwigLayer, self).__init__()
        self._callable_losses = []

    @property
    def losses(self):
        collected_losses = []
        for loss_fn in self._callable_losses:
            collected_losses.append(loss_fn())
        return collected_losses

    def add_losses(self, loss):
        if callable(loss):
            self._callable_losses.append(loss)

class LudwigModel(Module):
    def __init__(self):
        super(LudwigModel, self).__init__()

    @property
    def losses(self):
        collected_losses = []
        for layer in self.children():
            collected_losses.extend(layer.losses)
        return collected_losses

# I think I need this instead of what I have above:
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


class Dense(LudwigModule):
    def __init__(
        self,
        input_size,
        use_bias=True,
        weights_initializer='xavier_uniform',
        bias_initializer='zeros',
        weights_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
    ):
        super().__init__()
        self.dense = Linear(
            in_features=input_size,
            out_features=1,
            bias=use_bias
        )
        weights_initializer = initializers[weights_initializer]
        weights_initializer(self.dense.weight)

        bias_initializer = initializers[bias_initializer]
        bias_initializer(self.dense.bias)

        if weights_regularizer:
            self.add_loss(lambda: reg_loss(self.dense.weight, weights_regularizer))

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
            self.activation_loss = reg_loss(output, self.activity_regularizer) / batch_size
        return output
