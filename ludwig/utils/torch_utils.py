from torch.nn import Module
import torch
from torch.nn.init import (uniform_, normal_, constant_, ones_,  zeros_, eye_, dirac_,
        xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_)
from torch.nn import (ELU, LeakyReLU, LogSigmoid, ReLU, Sigmoid, Tanh, Softmax)

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
        super(LudwigLayer, self).__init__()
        self._callable_losses = []

    @property
    def losses(self):
        collected_losses = []
        for loss_fn in self._callable_losses:
            collected_losses.append(loss_fn())

        for child in self.children():
            collected_losses.extend(child.losses)


        return collected_losses

    def add_losses(self, loss):
        if callable(loss):
            self._callable_losses.append(loss)
