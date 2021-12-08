import torch
import torch.nn as nn


# taken from https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py
def glu(x: torch.Tensor, dim: int = -1):
    """Generalized linear unit nonlinear activation.

    Expects 2*n_units-dimensional input. Half of it is used to determine the gating of the GLU activation and the other
    half is used as an input to GLU,
    """
    return nn.functional.glu(x, dim)


def gelu(features: torch.Tensor, approximate: bool = False):
    if approximate:
        return 0.5 * features * (1.0 + nn.tanh(0.7978845608028654 * (features + 0.044715 * (features ** 3))))
    else:
        return 0.5 * features * (1.0 + torch.erf(features / 1.4142135623730951))
