import torch
import torch.nn as nn

from ludwig.schema.initializers import get_initialize_cls
from ludwig.utils.torch_utils import get_torch_device

DEVICE = "cuda:0" if get_torch_device() == "cuda" else "cpu"


def test_get_initialize_cls():
    """Currently only checks for when the parameters are default case."""
    # Initialization through the registry.
    tensor = torch.empty((2, 3), device=DEVICE)
    torch.random.manual_seed(0)
    initialized_tensor = get_initialize_cls("xavier_uniform")()(tensor)

    # Initialization through torch.
    default_initializer = nn.init.xavier_uniform_
    default_tensor = torch.empty((2, 3), device=DEVICE)
    torch.random.manual_seed(0)
    default_tensor = default_initializer(default_tensor)

    # Check that the tensor using the expected initialization and the same seed is identical
    assert torch.equal(initialized_tensor, default_tensor)
