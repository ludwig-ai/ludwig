import pytest
import torch
from torch.autograd import gradcheck

from ludwig.utils.entmax.activations import (
    _entmax_threshold_and_support,
    _sparsemax_threshold_and_support,
    Entmax15,
    Sparsemax,
)


@pytest.mark.parametrize("dim", (0, 1, 2))
@pytest.mark.parametrize("Map", (Sparsemax, Entmax15))
def test_mapping(dim, Map):
    f = Map(dim=dim, k=3)
    x = torch.randn(3, 4, 5, requires_grad=True, dtype=torch.float64)
    gradcheck(f, (x,))


@pytest.mark.parametrize("dim", (0, 1, 2))
@pytest.mark.parametrize("coef", (0.00001, 0.5, 10000))
def test_entmax_topk(dim, coef):
    x = coef * torch.randn(3, 4, 5)
    tau1, supp1 = _entmax_threshold_and_support(x, dim=dim, k=None)
    tau2, supp2 = _entmax_threshold_and_support(x, dim=dim, k=5)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)


@pytest.mark.parametrize("dim", (0, 1, 2))
@pytest.mark.parametrize("coef", (0.00001, 0.5, 10000))
@pytest.mark.parametrize("k", (5, 30))
def test_sparsemax_topk(dim, coef, k):
    x = coef * torch.randn(3, 4, 5)
    tau1, supp1 = _sparsemax_threshold_and_support(x, dim=dim, k=None)
    tau2, supp2 = _sparsemax_threshold_and_support(x, dim=dim, k=k)

    assert torch.all(tau1 == tau2)
    assert torch.all(supp1 == supp2)
