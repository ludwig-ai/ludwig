"""Phase 6.7.2 — Pareto-optimal multi-task loss balancer unit tests."""

from __future__ import annotations

import pytest
import torch

from ludwig.modules.loss_balancing import create_loss_balancer, LOSS_BALANCER_REGISTRY, ParetoMTLLossBalancer


class TestParetoMTLLossBalancer:
    def test_registered(self):
        assert "pareto_mtl" in LOSS_BALANCER_REGISTRY
        assert LOSS_BALANCER_REGISTRY["pareto_mtl"] is ParetoMTLLossBalancer

    def test_uniform_preference_default(self):
        balancer = ParetoMTLLossBalancer(["a", "b", "c"])
        torch.testing.assert_close(balancer.preference_vector, torch.tensor([1 / 3, 1 / 3, 1 / 3]))

    def test_explicit_preference_is_normalised(self):
        balancer = ParetoMTLLossBalancer(["a", "b"], preference_vector=[2.0, 3.0])
        torch.testing.assert_close(balancer.preference_vector, torch.tensor([0.4, 0.6]))

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="one per output feature"):
            ParetoMTLLossBalancer(["a", "b", "c"], preference_vector=[1.0, 1.0])

    def test_rejects_negative_entries(self):
        with pytest.raises(ValueError, match="non-negative"):
            ParetoMTLLossBalancer(["a", "b"], preference_vector=[1.0, -0.5])

    def test_rejects_zero_sum(self):
        with pytest.raises(ValueError, match="positive"):
            ParetoMTLLossBalancer(["a", "b"], preference_vector=[0.0, 0.0])

    def test_tchebycheff_weight_range(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            ParetoMTLLossBalancer(["a", "b"], tchebycheff_weight=1.5)

    def test_linear_scalarisation_at_tcheb_zero(self):
        """With tchebycheff_weight=0, the balancer is exactly sum(lam_i * L_i).

        Equal unit weights_i + losses (1.0, 2.0) with uniform preference (0.5, 0.5) gives
        0.5*1.0 + 0.5*2.0 = 1.5.
        """
        balancer = ParetoMTLLossBalancer(["a", "b"], preference_vector=[1.0, 1.0], tchebycheff_weight=0.0)
        per_task = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        weights = {"a": 1.0, "b": 1.0}
        out = balancer(per_task, weights)
        torch.testing.assert_close(out, torch.tensor(1.5))

    def test_tchebycheff_scalarisation_at_tcheb_one(self):
        """With tchebycheff_weight=1, the balancer is max(lam_i * L_i)."""
        balancer = ParetoMTLLossBalancer(["a", "b"], preference_vector=[1.0, 1.0], tchebycheff_weight=1.0)
        per_task = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        weights = {"a": 1.0, "b": 1.0}
        out = balancer(per_task, weights)
        # max(0.5*1.0, 0.5*2.0) = 1.0
        torch.testing.assert_close(out, torch.tensor(1.0))

    def test_mixed_scalarisation(self):
        balancer = ParetoMTLLossBalancer(["a", "b"], preference_vector=[1.0, 3.0], tchebycheff_weight=0.5)
        per_task = {"a": torch.tensor(4.0), "b": torch.tensor(2.0)}
        weights = {"a": 1.0, "b": 1.0}
        # Normalised preference = [0.25, 0.75].
        # linear term  = 0.25*4 + 0.75*2 = 1.0 + 1.5 = 2.5
        # tcheb   term = max(0.25*4, 0.75*2) = max(1.0, 1.5) = 1.5
        # blended      = 0.5*2.5 + 0.5*1.5 = 2.0
        out = balancer(per_task, weights)
        torch.testing.assert_close(out, torch.tensor(2.0))

    def test_backward_flows(self):
        balancer = ParetoMTLLossBalancer(["a", "b"], preference_vector=[1.0, 2.0])
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)
        out = balancer({"a": a, "b": b}, {"a": 1.0, "b": 1.0})
        out.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_create_loss_balancer_passthrough(self):
        balancer = create_loss_balancer(
            "pareto_mtl",
            ["a", "b"],
            preference_vector=[1.0, 4.0],
            tchebycheff_weight=0.25,
        )
        assert isinstance(balancer, ParetoMTLLossBalancer)
        torch.testing.assert_close(balancer.preference_vector, torch.tensor([0.2, 0.8]))
        assert balancer.tchebycheff_weight == 0.25


class TestParetoMTLSchema:
    def _base(self) -> dict:
        return {
            "input_features": [{"name": "a", "type": "number"}, {"name": "b", "type": "number"}],
            "output_features": [
                {"name": "y1", "type": "binary"},
                {"name": "y2", "type": "number"},
            ],
        }

    def test_pareto_mtl_config_accepted(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                **self._base(),
                "trainer": {
                    "loss_balancing": "pareto_mtl",
                    "loss_balancing_preference_vector": [1.0, 2.0],
                    "loss_balancing_tchebycheff_weight": 0.3,
                },
            }
        )
        assert cfg.trainer.loss_balancing == "pareto_mtl"
        assert cfg.trainer.loss_balancing_preference_vector == [1.0, 2.0]
        assert cfg.trainer.loss_balancing_tchebycheff_weight == 0.3

    def test_preference_vector_defaults_none(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(self._base())
        assert cfg.trainer.loss_balancing_preference_vector is None
        assert cfg.trainer.loss_balancing_tchebycheff_weight == 0.5
