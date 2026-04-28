#!/usr/bin/env python
"""Multi-Task Learning with Loss Balancing in Ludwig.

Trains four models on the UCI Wine Quality dataset with two output features:
  - quality_score  : raw 0-10 quality score (regression)
  - quality_binary : quality >= 7 is "good" (binary classification)

Compares loss balancing strategies:
  1. none         — static weighted sum (baseline)
  2. famo         — Fast Adaptive Multitask Optimization (available now)
  3. uncertainty  — Homoscedastic uncertainty weighting (available now)
  4. nash_mtl     — Nash bargaining solution (requires PR #4092)

# Colab: !pip install ludwig

Usage:
    python train_multi_task.py
"""

import logging
import os
import shutil
import warnings

import pandas as pd

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "wine-quality/winequality-red.csv"

WINE_FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_dataset() -> pd.DataFrame:
    """Download and prepare the dual-output wine quality dataset."""
    print("Downloading wine quality dataset...")
    df = pd.read_csv(WINE_URL, sep=";")
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # quality_score: keep the raw 0-10 numerical score
    df["quality_score"] = df["quality"].astype(float)
    # quality_binary: 1 if quality >= 7 (good wine), else 0
    df["quality_binary"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])
    print(f"  {len(df)} rows | good wines (quality >= 7): {df['quality_binary'].mean():.1%}")
    print(f"  quality_score range: {df['quality_score'].min():.0f} – {df['quality_score'].max():.0f}")
    return df


# ---------------------------------------------------------------------------
# Ludwig config helpers
# ---------------------------------------------------------------------------


def _input_features() -> list:
    return [{"name": feat, "type": "number", "preprocessing": {"normalization": "zscore"}} for feat in WINE_FEATURES]


def _base_config(loss_balancing: str) -> dict:
    return {
        "model_type": "ecd",
        "input_features": _input_features(),
        "output_features": [
            {"name": "quality_score", "type": "number"},
            {"name": "quality_binary", "type": "binary"},
        ],
        "combiner": {
            "type": "concat",
            "num_fc_layers": 2,
            "output_size": 128,
            "dropout": 0.1,
        },
        "trainer": {
            "epochs": 30,
            "learning_rate": 0.001,
            "batch_size": 128,
            "loss_balancing": loss_balancing,
        },
    }


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_and_evaluate(
    name: str,
    config: dict,
    dataset: pd.DataFrame,
    output_dir: str,
) -> dict | None:
    """Train a Ludwig model and return evaluation metrics.

    Returns a dict with metric values, or None if training failed.
    """
    from ludwig.api import LudwigModel

    result_dir = os.path.join(output_dir, name)
    shutil.rmtree(result_dir, ignore_errors=True)

    print(f"\n--- Training: {name} ---")
    try:
        model = LudwigModel(config=config, logging_level=logging.WARNING)
        result = model.train(
            dataset=dataset,
            experiment_name="multi_task",
            model_name=name,
            output_directory=result_dir,
        )

        # Extract final validation metrics
        metrics = {}
        vset = result.train_stats.validation or {}
        # quality_score: mean absolute error (lower is better)
        score_metrics = vset.get("quality_score", {})
        metrics["score_mae"] = _last_value(score_metrics.get("mean_absolute_error", []))
        # quality_binary: ROC AUC (higher is better)
        binary_metrics = vset.get("quality_binary", {})
        metrics["binary_roc_auc"] = _last_value(binary_metrics.get("roc_auc", []))
        return metrics

    except Exception as exc:
        warnings.warn(f"Training '{name}' failed: {exc}", stacklevel=2)
        return None


def _last_value(series) -> float | None:
    """Return the last numeric value in a list, or None."""
    if not series:
        return None
    val = series[-1]
    if isinstance(val, (list, tuple)):
        val = val[-1]
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(results: dict) -> None:
    """Print a formatted side-by-side comparison of all methods."""
    col_w = 14
    header = f"{'Method':<{col_w}} | " f"{'Score MAE':>{col_w}} | " f"{'Binary ROC-AUC':>{col_w}}"
    separator = "-" * len(header)
    print()
    print("=" * len(header))
    print("  Multi-Task Loss Balancing — Comparison")
    print("=" * len(header))
    print(header)
    print(separator)
    for method, metrics in results.items():
        if metrics is None:
            mae_str = "FAILED"
            auc_str = "FAILED"
        else:
            mae = metrics.get("score_mae")
            auc = metrics.get("binary_roc_auc")
            mae_str = f"{mae:.4f}" if mae is not None else "n/a"
            auc_str = f"{auc:.4f}" if auc is not None else "n/a"
        print(f"{method:<{col_w}} | {mae_str:>{col_w}} | {auc_str:>{col_w}}")
    print(separator)
    print("  Score MAE: lower is better | Binary ROC-AUC: higher is better")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset()

    # Methods to compare. nash_mtl is attempted but skipped gracefully if
    # PR #4092 is not yet merged.
    methods = [
        ("none", False),
        ("famo", False),
        ("uncertainty", False),
        ("nash_mtl", True),  # requires PR #4092
    ]

    results = {}
    for method, requires_pr in methods:
        if requires_pr:
            print(f"\n--- Skipping {method} (requires PR #4092 / Ludwig >= 0.14) ---")
            print("  To enable, install Ludwig from the 'future-capabilities' branch:")
            print("    pip install git+https://github.com/ludwig-ai/ludwig@future-capabilities")
            results[method] = None
            continue

        config = _base_config(method)
        results[method] = train_and_evaluate(method, config, df, output_dir)

    # Attempt nash_mtl — will succeed if PR #4092 is available
    try:
        from ludwig.api import LudwigModel  # noqa: F401

        config = _base_config("nash_mtl")
        # Try instantiating to check if nash_mtl is a valid option
        model = LudwigModel(config=config, logging_level=logging.WARNING)
        del model
        print("\n  nash_mtl is available — training now...")
        results["nash_mtl"] = train_and_evaluate("nash_mtl", config, df, output_dir)
    except Exception:
        pass  # already marked as None above

    print_comparison_table(results)


if __name__ == "__main__":
    main()
