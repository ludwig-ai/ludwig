"""Native Optuna hyperparameter optimization executor.

Provides direct Optuna integration without requiring Ray Tune as an intermediary.
Supports AutoSampler (auto-selects best algorithm), GPSampler (Bayesian optimization),
TPE, CMA-ES, and other Optuna samplers.

For distributed execution, use Ray Tune with OptunaSearch instead.

Usage:
    from ludwig.hyperopt.optuna_executor import OptunaExecutor

    executor = OptunaExecutor(
        parameters={"trainer.learning_rate": {"space": "loguniform", "lower": 1e-5, "upper": 1e-2}},
        metric="validation.combined.loss",
        goal="minimize",
        num_samples=50,
    )
    best_params = executor.optimize(train_fn)
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class OptunaExecutor:
    """Native Optuna hyperparameter optimization executor.

    Uses Optuna's define-by-run API for flexible search space definitions and efficient optimization algorithms.
    """

    def __init__(
        self,
        parameters: dict[str, dict[str, Any]],
        metric: str = "validation.combined.loss",
        goal: str = "minimize",
        num_samples: int = 20,
        sampler: str = "auto",
        pruner: str | None = None,
        study_name: str | None = None,
        storage: str | None = None,
    ):
        """Initialize the Optuna executor.

        Args:
            parameters: Dict mapping parameter paths to search space definitions.
                Each value is a dict with keys: space (str), lower/upper (float), etc.
                Supported spaces: uniform, loguniform, int, choice, categorical.
            metric: Metric to optimize (dot-separated path in results dict).
            goal: "minimize" or "maximize".
            num_samples: Number of trials to run.
            sampler: Sampler type: "auto" (AutoSampler), "gp" (GPSampler),
                "tpe" (TPE), "cmaes" (CMA-ES), "random" (RandomSampler).
            pruner: Optional pruner: "median", "hyperband", None.
            study_name: Name for the Optuna study (for persistence/resumption).
            storage: Optuna storage URL (e.g., "sqlite:///optuna.db") for persistence.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        self.parameters = parameters
        self.metric = metric
        self.goal = goal
        self.num_samples = num_samples

        # Create sampler
        sampler_obj = self._create_sampler(sampler)

        # Create pruner
        pruner_obj = None
        if pruner == "median":
            pruner_obj = optuna.pruners.MedianPruner()
        elif pruner == "hyperband":
            pruner_obj = optuna.pruners.HyperbandPruner()

        # Create study
        direction = "minimize" if goal == "minimize" else "maximize"
        self.study = optuna.create_study(
            study_name=study_name or "ludwig_hyperopt",
            direction=direction,
            sampler=sampler_obj,
            pruner=pruner_obj,
            storage=storage,
            load_if_exists=True,
        )

    def _create_sampler(self, sampler_type: str):
        """Create an Optuna sampler from type string."""
        import optuna

        if sampler_type == "auto":
            try:
                return optuna.samplers.AutoSampler()
            except AttributeError:
                # AutoSampler not available in older Optuna versions
                logger.info("AutoSampler not available, falling back to TPE")
                return optuna.samplers.TPESampler()
        elif sampler_type == "gp":
            try:
                return optuna.samplers.GPSampler()
            except AttributeError:
                logger.info("GPSampler not available, falling back to TPE")
                return optuna.samplers.TPESampler()
        elif sampler_type == "tpe":
            return optuna.samplers.TPESampler()
        elif sampler_type == "cmaes":
            return optuna.samplers.CmaEsSampler()
        elif sampler_type == "random":
            return optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Unknown sampler: {sampler_type}. Options: auto, gp, tpe, cmaes, random")

    def _suggest_params(self, trial) -> dict[str, Any]:
        """Suggest parameter values for a trial based on the search space definition."""
        params = {}
        for param_name, space_def in self.parameters.items():
            space_type = space_def.get("space", "uniform")
            if space_type == "uniform":
                params[param_name] = trial.suggest_float(param_name, space_def["lower"], space_def["upper"])
            elif space_type == "loguniform":
                params[param_name] = trial.suggest_float(param_name, space_def["lower"], space_def["upper"], log=True)
            elif space_type in ("int", "randint"):
                params[param_name] = trial.suggest_int(param_name, space_def["lower"], space_def["upper"])
            elif space_type in ("choice", "categorical"):
                params[param_name] = trial.suggest_categorical(param_name, space_def["categories"])
            elif space_type == "grid":
                params[param_name] = trial.suggest_categorical(param_name, space_def["values"])
            else:
                raise ValueError(f"Unknown search space type: {space_type} for parameter {param_name}")
        return params

    def optimize(self, objective_fn: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
        """Run the optimization.

        Args:
            objective_fn: Function that takes a parameter dict and returns the metric value.
                The function should train the model with the given parameters and return
                the validation metric.

        Returns:
            Dict with best parameters found.
        """

        def trial_fn(trial):
            params = self._suggest_params(trial)
            metric_value = objective_fn(params)
            return metric_value

        logger.info(f"Starting Optuna optimization: {self.num_samples} trials, {self.goal} {self.metric}")
        self.study.optimize(trial_fn, n_trials=self.num_samples)

        best = self.study.best_params
        logger.info(f"Optimization complete. Best {self.metric}: {self.study.best_value:.6f}")
        logger.info(f"Best parameters: {best}")

        return best

    def get_results(self) -> dict[str, Any]:
        """Get optimization results summary."""
        return {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "best_trial": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": str(t.state),
                }
                for t in self.study.trials
            ],
        }
