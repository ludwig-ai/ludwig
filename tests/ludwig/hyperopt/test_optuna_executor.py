"""Tests for native Optuna executor."""

import pytest

from ludwig.hyperopt.optuna_executor import _create_sampler, _suggest_params, OptunaExecutor


class TestOptunaExecutorInit:
    def test_constructor_matches_registry_interface(self):
        """OptunaExecutor must accept the same args as the executor registry passes."""
        executor = OptunaExecutor(
            parameters={"trainer.learning_rate": {"space": "loguniform", "lower": 1e-5, "upper": 1e-1}},
            output_feature="label",
            metric="accuracy",
            goal="maximize",
            split="validation",
            num_samples=5,
            sampler="tpe",
        )
        assert executor.output_feature == "label"
        assert executor.metric == "accuracy"
        assert executor.goal == "maximize"
        assert executor.num_samples == 5

    def test_registered_in_executor_registry(self):
        """OptunaExecutor must be available via executor_registry['optuna']."""
        from ludwig.hyperopt.execution import executor_registry

        assert "optuna" in executor_registry
        cls = executor_registry["optuna"]
        assert cls is OptunaExecutor

    def test_invalid_sampler_raises(self):
        with pytest.raises(ValueError, match="Unknown sampler"):
            _create_sampler("invalid")


class TestSuggestParams:
    def test_uniform(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(trial, {"x": {"space": "uniform", "lower": 0.0, "upper": 1.0}})
        assert 0.0 <= params["x"] <= 1.0

    def test_loguniform(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(trial, {"lr": {"space": "loguniform", "lower": 1e-5, "upper": 1e-1}})
        assert 1e-5 <= params["lr"] <= 1e-1

    def test_int(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(trial, {"n": {"space": "int", "lower": 1, "upper": 10}})
        assert isinstance(params["n"], int)
        assert 1 <= params["n"] <= 10

    def test_categorical(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(trial, {"opt": {"space": "choice", "categories": ["adam", "sgd"]}})
        assert params["opt"] in ["adam", "sgd"]

    def test_grid_search(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(trial, {"bs": {"space": "grid_search", "values": [32, 64, 128]}})
        assert params["bs"] in [32, 64, 128]

    def test_invalid_space_raises(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(ValueError, match="Unknown search space"):
            _suggest_params(trial, {"x": {"space": "unknown", "lower": 0, "upper": 1}})

    def test_multiple_params(self):
        import optuna

        study = optuna.create_study()
        trial = study.ask()
        params = _suggest_params(
            trial,
            {
                "lr": {"space": "loguniform", "lower": 1e-5, "upper": 1e-1},
                "bs": {"space": "int", "lower": 16, "upper": 256},
                "opt": {"space": "choice", "categories": ["adam", "sgd"]},
            },
        )
        assert "lr" in params
        assert "bs" in params
        assert "opt" in params


class TestSamplers:
    @pytest.mark.parametrize("sampler_type", ["auto", "tpe", "random", "cmaes"])
    def test_create_sampler(self, sampler_type):
        sampler = _create_sampler(sampler_type)
        assert sampler is not None
