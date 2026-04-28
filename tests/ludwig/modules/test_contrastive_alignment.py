"""Unit tests for ContrastiveAlignmentLoss (Phase 6.4.2)."""

from __future__ import annotations

import math

import pytest
import torch

from ludwig.modules.contrastive_alignment import ContrastiveAlignmentLoss


class TestContrastiveAlignmentLoss:
    def test_shape_and_scalar(self):
        torch.manual_seed(0)
        loss_fn = ContrastiveAlignmentLoss({"a": 8, "b": 12}, projection_dim=16)
        batch = {
            "a": torch.randn(4, 8),
            "b": torch.randn(4, 12),
        }
        loss = loss_fn(batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_perfect_alignment_near_zero_loss(self):
        """When two feature embeddings are identical and projection is identity-ish, loss should be very small
        because the diagonal dominates the similarity matrix."""
        torch.manual_seed(0)
        # Use a large projection_dim and initialize projections to near-identity so paired
        # examples map to near-identical points in the aligned space.
        loss_fn = ContrastiveAlignmentLoss({"a": 16, "b": 16}, projection_dim=16, learnable_temperature=False)
        with torch.no_grad():
            for name in ["a", "b"]:
                loss_fn.projections[name].weight.copy_(torch.eye(16))
                loss_fn.projections[name].bias.zero_()
        x = torch.randn(32, 16)
        loss = loss_fn({"a": x, "b": x.clone()})
        # With identical features and an identity projection the contrastive loss is minimised.
        assert loss.item() < 0.1

    def test_misalignment_increases_loss(self):
        """Pairing example i with i works better than random pairing."""
        torch.manual_seed(0)
        loss_fn = ContrastiveAlignmentLoss({"a": 16, "b": 16}, projection_dim=16, learnable_temperature=False)
        with torch.no_grad():
            for name in ["a", "b"]:
                loss_fn.projections[name].weight.copy_(torch.eye(16))
                loss_fn.projections[name].bias.zero_()
        x = torch.randn(32, 16)

        aligned = loss_fn({"a": x, "b": x.clone()})
        # Shuffle feature b so its positive is no longer at position i.
        perm = torch.randperm(32)
        misaligned = loss_fn({"a": x, "b": x[perm].clone()})
        assert misaligned.item() > aligned.item()

    def test_pair_symmetry(self):
        """Swapping two feature names should leave the loss value unchanged (up to floating-point)."""
        torch.manual_seed(0)
        loss_fn = ContrastiveAlignmentLoss({"a": 8, "b": 8}, projection_dim=16, learnable_temperature=False)
        with torch.no_grad():
            # Copy feature-a projection weights into feature-b so the two features become
            # interchangeable.
            loss_fn.projections["b"].weight.copy_(loss_fn.projections["a"].weight)
            loss_fn.projections["b"].bias.copy_(loss_fn.projections["a"].bias)
        x = torch.randn(16, 8)
        y = torch.randn(16, 8)
        ab = loss_fn({"a": x, "b": y})
        ba = loss_fn({"a": y, "b": x})
        assert torch.allclose(ab, ba, atol=1e-5)

    def test_three_features(self):
        """Loss should accommodate any number >= 2 of features and average over pairs."""
        torch.manual_seed(0)
        loss_fn = ContrastiveAlignmentLoss({"a": 4, "b": 5, "c": 6}, projection_dim=8)
        embeddings = {
            "a": torch.randn(4, 4),
            "b": torch.randn(4, 5),
            "c": torch.randn(4, 6),
        }
        loss = loss_fn(embeddings)
        assert torch.isfinite(loss)

    def test_rejects_single_feature(self):
        with pytest.raises(ValueError, match="at least 2 input features"):
            ContrastiveAlignmentLoss({"only_one": 8})

    def test_rejects_missing_feature_in_batch(self):
        loss_fn = ContrastiveAlignmentLoss({"a": 4, "b": 4}, projection_dim=8)
        with pytest.raises(KeyError, match="expected feature 'b'"):
            loss_fn({"a": torch.randn(2, 4)})

    def test_learnable_vs_fixed_temperature(self):
        fixed = ContrastiveAlignmentLoss({"a": 4, "b": 4}, projection_dim=8, learnable_temperature=False)
        learnable = ContrastiveAlignmentLoss({"a": 4, "b": 4}, projection_dim=8, learnable_temperature=True)

        assert not fixed.log_temperature.requires_grad
        assert learnable.log_temperature.requires_grad
        # Both should start at log(1/0.07).
        expected = math.log(1.0 / 0.07)
        assert abs(float(fixed.log_temperature) - expected) < 1e-5
        assert abs(float(learnable.log_temperature) - expected) < 1e-5

    def test_backward_populates_encoder_grads(self):
        """The loss gradient must flow into the per-feature encoder inputs so an upstream encoder is actually
        updated during pre-alignment."""
        loss_fn = ContrastiveAlignmentLoss({"a": 8, "b": 8}, projection_dim=16)
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(4, 8, requires_grad=True)
        loss = loss_fn({"a": a, "b": b})
        loss.backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert b.grad is not None and torch.isfinite(b.grad).all()


class TestContrastivePretrainSchema:
    def test_default_values(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "input_features": [
                    {"name": "a", "type": "number"},
                    {"name": "b", "type": "number"},
                ],
                "output_features": [{"name": "y", "type": "binary"}],
            }
        )
        assert cfg.trainer.contrastive_pretrain_epochs == 0
        assert cfg.trainer.contrastive_pretrain_temperature == 0.07
        assert cfg.trainer.contrastive_pretrain_projection_dim == 128

    def test_explicit_values(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "input_features": [
                    {"name": "a", "type": "number"},
                    {"name": "b", "type": "number"},
                ],
                "output_features": [{"name": "y", "type": "binary"}],
                "trainer": {
                    "contrastive_pretrain_epochs": 3,
                    "contrastive_pretrain_temperature": 0.1,
                    "contrastive_pretrain_projection_dim": 64,
                },
            }
        )
        assert cfg.trainer.contrastive_pretrain_epochs == 3
        assert cfg.trainer.contrastive_pretrain_temperature == 0.1
        assert cfg.trainer.contrastive_pretrain_projection_dim == 64
