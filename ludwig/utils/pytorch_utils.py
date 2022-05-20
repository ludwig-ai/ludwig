from torch import nn


def freeze_parameters(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
