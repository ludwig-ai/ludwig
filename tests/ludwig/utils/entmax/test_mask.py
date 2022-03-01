import torch
import pytest

from ludwig.utils.entmax.activations import Sparsemax, Entmax15

from ludwig.utils.entmax.root_finding import sparsemax_bisect, entmax_bisect

funcs = [
    Sparsemax(dim=1),
    Entmax15(dim=1),
    Sparsemax(dim=1, k=512),
    Entmax15(dim=1, k=512),
    sparsemax_bisect,
    entmax_bisect,
]


@pytest.mark.parametrize("func", funcs)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mask(func, dtype):
    torch.manual_seed(42)
    x = torch.randn(2, 6, dtype=dtype)
    x[:, 3:] = -float("inf")
    x0 = x[:, :3]

    y = func(x)
    y0 = func(x0)

    y[:, :3] -= y0

    assert torch.allclose(y, torch.zeros_like(y))


@pytest.mark.parametrize("alpha", (1.25, 1.5, 1.75, 2.25))
def test_mask_alphas(alpha):
    torch.manual_seed(42)
    x = torch.randn(2, 6)
    x[:, 3:] = -float("inf")
    x0 = x[:, :3]

    y = entmax_bisect(x, alpha)
    y0 = entmax_bisect(x0, alpha)

    y[:, :3] -= y0

    assert torch.allclose(y, torch.zeros_like(y))
