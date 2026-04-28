"""Tests for BatchEnsemble."""

import torch

from ludwig.modules.batch_ensemble import BatchEnsembleLinear


class TestBatchEnsembleLinear:
    def test_output_shape(self):
        layer = BatchEnsembleLinear(128, 64, num_members=4)
        x = torch.randn(16, 128)
        out = layer(x)
        assert out.shape == (16, 64)

    def test_train_vs_eval(self):
        layer = BatchEnsembleLinear(32, 16, num_members=4)
        x = torch.randn(8, 32)

        layer.train()
        out_train = layer(x)

        layer.eval()
        out_eval = layer(x)

        assert out_train.shape == out_eval.shape

    def test_gradient_flow(self):
        layer = BatchEnsembleLinear(32, 16, num_members=4)
        layer.train()
        x = torch.randn(8, 32, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.r.grad is not None
        assert layer.s.grad is not None

    def test_per_member_params(self):
        layer = BatchEnsembleLinear(32, 16, num_members=4)
        assert layer.r.shape == (4, 32)
        assert layer.s.shape == (4, 16)

    def test_no_bias(self):
        layer = BatchEnsembleLinear(32, 16, num_members=2, bias=False)
        assert layer.bias is None
        x = torch.randn(4, 32)
        out = layer(x)
        assert out.shape == (4, 16)
