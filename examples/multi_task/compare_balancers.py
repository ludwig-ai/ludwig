"""Compare multi-task loss balancing strategies on a joint classification + regression task.

The dataset is UCI Wine Quality (red) with two output features:

- ``quality`` — the usual 0–10 score, trained as number regression.
- ``recommended`` — a synthetic binary target set to ``quality >= 6``, trained as binary
  classification. The two outputs share everything except the final decoder head, so they
  compete for the combiner's representational capacity.

For each balancer in :data:`STRATEGIES` the script trains the same model end-to-end and
records validation metrics. The summary table prints the per-task scores plus a
balance-aware geometric mean so you can see which strategy gets both tasks right.

Requires Ludwig 0.15 / PR #4092 for ``nash_mtl``.

Run: ``python compare_balancers.py``
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import wine_quality

HERE = Path(__file__).parent

# Strategies to compare. nash_mtl is included only on the future-capabilities branch.
STRATEGIES = [
    "none",
    "log_transform",
    "uncertainty",
    "famo",
    "gradnorm",
    "nash_mtl",
]


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recommended"] = (df["quality"] >= 6).astype(int)
    return df


def build_config(balancer: str) -> dict:
    with (HERE / "config_nash_mtl.yaml").open() as f:
        config = yaml.safe_load(f)
    config["trainer"]["loss_balancing"] = balancer
    return config


def run(balancer: str, dataset: pd.DataFrame) -> dict[str, float]:
    config = build_config(balancer)
    model = LudwigModel(config=config, logging_level=logging.WARNING)
    result = model.train(
        dataset=dataset,
        output_directory=str(HERE / f"results_{balancer}"),
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_model=True,
    )
    val = result.train_stats.validation or {}

    quality_rmse = min(val["quality"].get("root_mean_squared_error", [float("nan")]))
    recommended_acc = max(val["recommended"].get("accuracy", [float("nan")]))
    quality_loss = min(val["quality"].get("loss", [float("nan")]))
    recommended_loss = min(val["recommended"].get("loss", [float("nan")]))

    # Geometric mean of losses is a balance-aware aggregate: a strategy that wrecks one task
    # to win the other pays more than a strategy that keeps both reasonable.
    geomean = math.sqrt(quality_loss * recommended_loss) if quality_loss and recommended_loss else float("nan")
    return {
        "quality_rmse": quality_rmse,
        "recommended_acc": recommended_acc,
        "geomean_loss": geomean,
    }


def main() -> None:
    dataset = add_binary_target(wine_quality.load())

    rows = []
    for balancer in STRATEGIES:
        print(f"\n=== Training with loss_balancing: {balancer} ===")
        try:
            scores = run(balancer, dataset)
        except Exception as exc:
            print(f"[skip] {balancer}: {exc}")
            continue
        rows.append({"balancer": balancer, **scores})

    if not rows:
        raise SystemExit("No balancer runs completed successfully.")

    summary = pd.DataFrame(rows).set_index("balancer")
    summary = summary.sort_values("geomean_loss")
    print("\nResults (best-of-training per task, sorted by geomean_loss):")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    csv_path = HERE / "balancer_comparison.csv"
    summary.to_csv(csv_path)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
