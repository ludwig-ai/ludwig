import torch
import torch.nn as nn

from ludwig.modules.initializer_modules import get_initializer
from ludwig.utils.torch_utils import get_torch_device

DEVICE = "cuda:0" if get_torch_device() == "cuda" else "cpu"


def test_get_initializer():
    """Currently only checks for when the parameters are default case."""
    tensor_size = (2, 3)

    # Test for when the parameters are default
    torch.random.manual_seed(0)
    initialized_tensor = get_initializer("xavier_uniform")(*tensor_size, device=DEVICE)

    # Check that the tensor using the expected initialization and the same seed is identical
    default_initializer = nn.init.xavier_uniform_
    torch.random.manual_seed(0)
    default_tensor = default_initializer(torch.empty(*tensor_size, device=DEVICE))
    assert torch.equal(initialized_tensor, default_tensor)
