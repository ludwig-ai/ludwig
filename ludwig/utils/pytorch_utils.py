from torch import nn


def freeze_parameters(module: nn.Module):
    """Freezes the parameters of a torch module."""
    for p in module.parameters():
        p.requires_grad = False
