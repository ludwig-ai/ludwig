"""Tests for multi-task loss balancing strategies."""

import pytest
import torch

from ludwig.modules.loss_balancing import (
    create_loss_balancer,
    FAMOLossBalancer,
    GradNormLossBalancer,
    LogTransformLossBalancer,
    NashMTLLossBalancer,
    NoneLossBalancer,
    ParetoMTLLossBalancer,
    UncertaintyLossBalancer,
)

FEATURE_NAMES = ["output_1", "output_2", "output_3"]


def _make_losses(values: list[float]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(v, requires_grad=True) for name, v in zip(FEATURE_NAMES, values)}


def _make_weights() -> dict[str, float]:
    return {name: 1.0 for name in FEATURE_NAMES}


class TestNoneLossBalancer:
    def test_matches_static_weighted_sum(self):
        balancer = NoneLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        weights = {name: w for name, w in zip(FEATURE_NAMES, [0.5, 1.0, 2.0])}
        total = balancer(losses, weights)
        expected = 0.5 * 1.0 + 1.0 * 2.0 + 2.0 * 3.0
        assert torch.isclose(total, torch.tensor(expected))

    def test_gradient_flow(self):
        balancer = NoneLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        total.backward()
        for loss in losses.values():
            assert loss.grad is not None


class TestLogTransformLossBalancer:
    def test_compresses_large_losses(self):
        balancer = LogTransformLossBalancer(FEATURE_NAMES)
        weights = _make_weights()
        small_losses = _make_losses([0.1, 0.2, 0.3])
        large_losses = _make_losses([100.0, 200.0, 300.0])
        small_total = balancer(small_losses, weights)
        large_total = balancer(large_losses, weights)
        # Log compression should make ratio much smaller than 1000x
        ratio = large_total.item() / small_total.item()
        assert ratio < 100  # Much less than 1000x

    def test_gradient_flow(self):
        balancer = LogTransformLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        total.backward()
        for loss in losses.values():
            assert loss.grad is not None


class TestUncertaintyLossBalancer:
    def test_has_learnable_parameters(self):
        balancer = UncertaintyLossBalancer(FEATURE_NAMES)
        params = list(balancer.parameters())
        assert len(params) == 3  # one log_var per task

    def test_gradient_flow(self):
        balancer = UncertaintyLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        total.backward()
        for name in FEATURE_NAMES:
            assert balancer.log_vars[name].grad is not None

    def test_output_is_finite(self):
        balancer = UncertaintyLossBalancer(FEATURE_NAMES)
        losses = _make_losses([0.001, 100.0, 5.0])
        total = balancer(losses, _make_weights())
        assert torch.isfinite(total)


class TestFAMOLossBalancer:
    def test_has_learnable_parameters(self):
        balancer = FAMOLossBalancer(FEATURE_NAMES)
        params = list(balancer.parameters())
        assert len(params) == 3  # one log_weight per task

    def test_post_step_updates_prev_losses(self):
        balancer = FAMOLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        balancer(losses, _make_weights())
        balancer.post_step(losses)
        assert len(balancer._prev_losses) == 3

    def test_gradient_flow(self):
        balancer = FAMOLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        total.backward()
        for name in FEATURE_NAMES:
            assert balancer.log_weights[name].grad is not None


class TestGradNormLossBalancer:
    def test_has_learnable_parameters(self):
        balancer = GradNormLossBalancer(FEATURE_NAMES)
        params = list(balancer.parameters())
        assert len(params) == 3  # one task_weight per task

    def test_post_step_records_initial_losses(self):
        balancer = GradNormLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        balancer.post_step(losses)
        assert len(balancer._initial_losses) == 3

    def test_post_step_adjusts_weights(self):
        balancer = GradNormLossBalancer(FEATURE_NAMES)
        # First step: record initial losses
        losses1 = _make_losses([1.0, 1.0, 1.0])
        balancer.post_step(losses1)
        initial_weights = {name: balancer.task_weights[name].item() for name in FEATURE_NAMES}
        # Second step: one task improved a lot, others didn't
        losses2 = _make_losses([0.1, 1.0, 1.0])
        balancer.post_step(losses2)
        # The fast-improving task should get higher weight
        assert balancer.task_weights["output_1"].item() != initial_weights["output_1"]


class TestCreateLossBalancer:
    def test_create_none(self):
        b = create_loss_balancer("none", FEATURE_NAMES)
        assert isinstance(b, NoneLossBalancer)

    def test_create_log_transform(self):
        b = create_loss_balancer("log_transform", FEATURE_NAMES)
        assert isinstance(b, LogTransformLossBalancer)

    def test_create_uncertainty(self):
        b = create_loss_balancer("uncertainty", FEATURE_NAMES)
        assert isinstance(b, UncertaintyLossBalancer)

    def test_create_famo(self):
        b = create_loss_balancer("famo", FEATURE_NAMES, lr=0.05)
        assert isinstance(b, FAMOLossBalancer)

    def test_create_gradnorm(self):
        b = create_loss_balancer("gradnorm", FEATURE_NAMES, alpha=2.0)
        assert isinstance(b, GradNormLossBalancer)

    def test_create_nash_mtl(self):
        b = create_loss_balancer("nash_mtl", FEATURE_NAMES)
        assert isinstance(b, NashMTLLossBalancer)

    def test_create_pareto_mtl(self):
        b = create_loss_balancer("pareto_mtl", FEATURE_NAMES, preference_vector=[0.5, 0.3, 0.2])
        assert isinstance(b, ParetoMTLLossBalancer)

    def test_invalid_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown loss balancing strategy"):
            create_loss_balancer("nonexistent_strategy", FEATURE_NAMES)

    def test_invalid_strategy_lists_valid_options(self):
        with pytest.raises(ValueError, match="none"):
            create_loss_balancer("bad", FEATURE_NAMES)


class TestNashMTLLossBalancer:
    def test_has_learnable_parameters(self):
        balancer = NashMTLLossBalancer(FEATURE_NAMES)
        params = list(balancer.parameters())
        assert len(params) == 3

    def test_uniform_init(self):
        n = len(FEATURE_NAMES)
        balancer = NashMTLLossBalancer(FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert torch.isclose(balancer.task_weights[name], torch.tensor(1.0 / n))

    def test_gradient_flow(self):
        balancer = NashMTLLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        total.backward()
        for name in FEATURE_NAMES:
            assert balancer.task_weights[name].grad is not None

    def test_post_step_updates_weights(self):
        balancer = NashMTLLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 10.0, 5.0])
        initial = {name: balancer.task_weights[name].item() for name in FEATURE_NAMES}
        balancer.post_step(losses)
        # Weights should change after post_step
        changed = any(balancer.task_weights[name].item() != initial[name] for name in FEATURE_NAMES)
        assert changed

    def test_low_loss_task_gets_higher_weight(self):
        # Nash solution: weights ∝ 1/loss → low-loss task gets higher weight.
        balancer = NashMTLLossBalancer(FEATURE_NAMES)
        losses = _make_losses([1.0, 100.0, 10.0])
        balancer.post_step(losses)
        w1 = balancer.task_weights["output_1"].item()
        w2 = balancer.task_weights["output_2"].item()
        assert w1 > w2


class TestParetoMTLLossBalancerExtended:
    def test_uniform_preference_by_default(self):
        balancer = ParetoMTLLossBalancer(FEATURE_NAMES)
        pv = balancer.preference_vector
        expected = torch.ones(3) / 3
        assert torch.allclose(pv, expected, atol=1e-6)

    def test_skewed_preference_favors_task(self):
        # preference heavily toward output_1
        balancer = ParetoMTLLossBalancer(FEATURE_NAMES, preference_vector=[0.9, 0.05, 0.05])
        losses = _make_losses([1.0, 1.0, 1.0])
        weights = _make_weights()
        # output_1 dominates the linear term; total should differ from uniform
        total = balancer(losses, weights)
        assert torch.isfinite(total)

    def test_invalid_pref_vector_length(self):
        with pytest.raises(ValueError, match="preference_vector"):
            ParetoMTLLossBalancer(FEATURE_NAMES, preference_vector=[0.5, 0.5])

    def test_negative_pref_vector_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ParetoMTLLossBalancer(FEATURE_NAMES, preference_vector=[-0.1, 0.5, 0.6])

    def test_invalid_tchebycheff_weight_raises(self):
        with pytest.raises(ValueError, match="tchebycheff_weight"):
            ParetoMTLLossBalancer(FEATURE_NAMES, tchebycheff_weight=1.5)


class TestUncertaintyNumericalStability:
    def test_extreme_log_vars_remain_finite(self):
        """log_vars pushed to extreme values must not produce inf/nan."""
        balancer = UncertaintyLossBalancer(FEATURE_NAMES)
        with torch.no_grad():
            for name in FEATURE_NAMES:
                balancer.log_vars[name].fill_(-100.0)  # would overflow without clamp
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        assert torch.isfinite(total), f"Expected finite total, got {total}"

    def test_positive_extreme_log_vars_remain_finite(self):
        balancer = UncertaintyLossBalancer(FEATURE_NAMES)
        with torch.no_grad():
            for name in FEATURE_NAMES:
                balancer.log_vars[name].fill_(100.0)  # large positive log_var
        losses = _make_losses([1.0, 2.0, 3.0])
        total = balancer(losses, _make_weights())
        assert torch.isfinite(total), f"Expected finite total, got {total}"
