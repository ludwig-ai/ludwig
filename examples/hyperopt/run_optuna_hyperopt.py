"""Native Optuna hyperparameter optimization on Wine Quality.

Requires Ludwig 0.15 / PR #4090 (``ludwig[hyperopt]``, which pulls in ``optuna``).

Run: ``python run_optuna_hyperopt.py``

To persist trials and resume across restarts, edit ``config_optuna.yaml`` and set

.. code:: yaml

    hyperopt:
      executor:
        storage: sqlite:///wine_hpo.db

Then run the script again — Optuna will keep successful trials and continue sampling.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

from ludwig.datasets import wine_quality
from ludwig.hyperopt.run import hyperopt

HERE = Path(__file__).parent


def main() -> None:
    with (HERE / "config_optuna.yaml").open() as f:
        config = yaml.safe_load(f)

    dataset = wine_quality.load()

    print("Starting native-Optuna hyperopt on Wine Quality…")
    t0 = time.time()
    results = hyperopt(
        config=config,
        dataset=dataset,
        output_directory=str(HERE / "results_optuna"),
        experiment_name="wine_optuna",
        model_name="run",
        logging_level=logging.WARNING,
    )
    elapsed = time.time() - t0

    # ``results`` exposes an ``ordered_trials`` list with (metric, params, ...) tuples.
    best = results.ordered_trials[0]
    print("\n[Optuna] Best trial:")
    print(f"  metric value: {best.metric_score:.4f}")
    print("  params:")
    for k, v in best.parameters.items():
        print(f"    {k}: {v}")
    print(f"  completed in: {elapsed:.1f}s over {len(results.ordered_trials)} trials")


if __name__ == "__main__":
    main()
