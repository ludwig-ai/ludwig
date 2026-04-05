"""Tests for native Optuna executor."""

import pytest

from ludwig.hyperopt.optuna_executor import OptunaExecutor


class TestOptunaExecutor:
    def test_basic_optimization(self):
        """Test that Optuna finds good parameters for a simple quadratic."""
        params = {
            "x": {"space": "uniform", "lower": -10.0, "upper": 10.0},
        }
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=20)

        def objective(p):
            return (p["x"] - 3.0) ** 2  # minimum at x=3

        best = executor.optimize(objective)
        assert abs(best["x"] - 3.0) < 2.0  # Should be close to 3

    def test_maximize(self):
        params = {"x": {"space": "uniform", "lower": 0.0, "upper": 1.0}}
        executor = OptunaExecutor(parameters=params, goal="maximize", num_samples=15)

        def objective(p):
            return -((p["x"] - 0.7) ** 2)  # maximum at x=0.7

        best = executor.optimize(objective)
        assert abs(best["x"] - 0.7) < 0.3

    def test_loguniform_space(self):
        params = {"lr": {"space": "loguniform", "lower": 1e-5, "upper": 1e-1}}
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=10)
        best = executor.optimize(lambda p: p["lr"])  # minimize lr
        assert best["lr"] < 0.01

    def test_int_space(self):
        params = {"n_layers": {"space": "int", "lower": 1, "upper": 10}}
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=10)
        best = executor.optimize(lambda p: abs(p["n_layers"] - 3))
        assert isinstance(best["n_layers"], int)

    def test_categorical_space(self):
        params = {"optimizer": {"space": "choice", "categories": ["adam", "sgd", "adamw"]}}
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=10)
        best = executor.optimize(lambda p: {"adam": 0.1, "sgd": 0.5, "adamw": 0.2}[p["optimizer"]])
        assert best["optimizer"] in ["adam", "sgd", "adamw"]

    def test_multiple_params(self):
        params = {
            "lr": {"space": "loguniform", "lower": 1e-5, "upper": 1e-1},
            "batch_size": {"space": "int", "lower": 16, "upper": 256},
        }
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=10)
        best = executor.optimize(lambda p: p["lr"] + p["batch_size"] / 1000)
        assert "lr" in best
        assert "batch_size" in best

    def test_get_results(self):
        params = {"x": {"space": "uniform", "lower": 0.0, "upper": 1.0}}
        executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=5)
        executor.optimize(lambda p: p["x"] ** 2)
        results = executor.get_results()
        assert "best_params" in results
        assert "best_value" in results
        assert results["n_trials"] == 5
        assert len(results["trials"]) == 5

    def test_samplers(self):
        params = {"x": {"space": "uniform", "lower": 0.0, "upper": 1.0}}
        for sampler in ["auto", "tpe", "random"]:
            executor = OptunaExecutor(parameters=params, goal="minimize", num_samples=3, sampler=sampler)
            executor.optimize(lambda p: p["x"])

    def test_invalid_sampler_raises(self):
        params = {"x": {"space": "uniform", "lower": 0.0, "upper": 1.0}}
        with pytest.raises(ValueError, match="Unknown sampler"):
            OptunaExecutor(parameters=params, sampler="invalid")

    def test_invalid_space_raises(self):
        params = {"x": {"space": "unknown_space", "lower": 0.0, "upper": 1.0}}
        executor = OptunaExecutor(parameters=params, num_samples=1)
        with pytest.raises(ValueError, match="Unknown search space"):
            executor.optimize(lambda p: 0)
