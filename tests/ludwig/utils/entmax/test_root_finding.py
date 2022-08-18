from functools import partial
from itertools import product

import pytest
import torch
from torch.autograd import gradcheck

from ludwig.utils.entmax.activations import entmax15, sparsemax
from ludwig.utils.entmax.root_finding import entmax_bisect, sparsemax_bisect

# @pytest.mark.parametrize("dim", (0, 1, 2))
# def test_dim(dim, Map):
# for _ in range(10):
# x = torch.randn(5, 6, 7, requires_grad=True, dtype=torch.float64)
# # gradcheck(f, (x,))


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("bisect_training", [True, False])
def test_sparsemax(training, bisect_training):
    x = 0.5 * torch.randn(4, 6, dtype=torch.float32)
    p1 = sparsemax(x, 1, training=training)
    p2 = sparsemax_bisect(x, training=bisect_training)
    assert torch.sum((p1 - p2) ** 2) < 1e-7


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("bisect_training", [True, False])
def test_entmax15(training, bisect_training):
    x = 0.5 * torch.randn(4, 6, dtype=torch.float32)
    p1 = entmax15(x, 1, training=training)
    p2 = entmax_bisect(x, alpha=1.5, training=bisect_training)
    assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_sparsemax_grad():
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(sparsemax_bisect, (x,), eps=1e-5)


@pytest.mark.parametrize("alpha", (0.2, 0.5, 0.75, 1.2, 1.5, 1.75, 2.25))
def test_entmax_grad(alpha):
    alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(entmax_bisect, (x, alpha), eps=1e-5)


def test_entmax_correct_multiple_alphas():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 0.05 + 2.5 * torch.rand((n, 1), dtype=torch.float64, requires_grad=True)

    p1 = entmax_bisect(x, alpha)
    p2_ = [entmax_bisect(x[i].unsqueeze(0), alpha[i].item()).squeeze() for i in range(n)]
    p2 = torch.stack(p2_)

    assert torch.allclose(p1, p2)


def test_entmax_grad_multiple_alphas():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 0.05 + 2.5 * torch.rand((n, 1), dtype=torch.float64, requires_grad=True)
    gradcheck(entmax_bisect, (x, alpha), eps=1e-5)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]
    X = torch.randn(*shape, dtype=torch.float64)

    alpha_shape = shape
    alpha_shape[dim] = 1

    alphas = 0.05 + 2.5 * torch.rand(alpha_shape, dtype=torch.float64)

    P = entmax_bisect(X, alpha=alphas, dim=dim)

    ranges = [list(range(k)) if i != dim else [slice(None)] for i, k in enumerate(shape)]

    for ix in product(*ranges):
        x = X[ix].unsqueeze(0)
        alpha = alphas[ix].item()
        p_true = entmax_bisect(x, alpha=alpha, dim=-1)
        assert torch.allclose(P[ix], p_true)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_arbitrary_dimension_grad(dim):
    shape = [3, 4, 2, 5]

    alpha_shape = shape
    alpha_shape[dim] = 1

    f = partial(entmax_bisect, dim=dim)

    X = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    alphas = 0.05 + 2.5 * torch.rand(alpha_shape, dtype=torch.float64, requires_grad=True)
    gradcheck(f, (X, alphas), eps=1e-5)
