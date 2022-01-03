import torch
import torch.nn as nn

from ludwig.modules.initializer_modules import get_initializer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_get_initializer():
    """Currently only checks for when the parameters are None."""
    tensor_size = (2, 3)

    # Test for when the parameters are None
    torch.random.manual_seed(0)
    initialized_tensor = get_initializer(None)(*tensor_size, device=DEVICE)

    # Check that the tensor using the expected initialization and the same seed is identical
    default_initializer = nn.init.xavier_uniform_
    torch.random.manual_seed(0)
    default_tensor = default_initializer(torch.empty(*tensor_size, device=DEVICE))
    assert torch.equal(initialized_tensor, default_tensor)
