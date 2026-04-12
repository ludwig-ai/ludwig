"""
Hyperparameter Optimization with Native Optuna Executor
========================================================

NOTE: Requires PR #4090 to be merged, or Ludwig >= 0.14.
      Install dependencies: pip install ludwig optuna

Usage:
    python optuna_executor.py

The script downloads the UCI Wine Quality dataset, binarises the target
(quality >= 7), and runs Ludwig HPO using the native Optuna executor.
Results are persisted in `optuna_results.db` so interrupted runs can
be resumed by simply re-running the script.
"""

# Colab: !pip install ludwig optuna --quiet

import pathlib
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Download dataset
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "wine-quality/winequality-white.csv"
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/" "wine-quality/winequality-red.csv"

white_path = DATA_DIR / "winequality-white.csv"
red_path = DATA_DIR / "winequality-red.csv"
combined_path = DATA_DIR / "wine_quality.csv"

if not combined_path.exists():
    print("Downloading Wine Quality dataset from UCI …")
    urllib.request.urlretrieve(WHITE_URL, white_path)
    urllib.request.urlretrieve(RED_URL, red_path)

    white = pd.read_csv(white_path, sep=";")
    red = pd.read_csv(red_path, sep=";")
    df = pd.concat([white, red], ignore_index=True)

    # Binary target: 1 if quality >= 7, else 0
    df["quality"] = (df["quality"] >= 7).astype(int)

    df.to_csv(combined_path, index=False)
    print(f"Dataset saved to {combined_path} ({len(df)} rows)")
else:
    print(f"Dataset already present at {combined_path}")
    df = pd.read_csv(combined_path)
    print(f"  {len(df)} rows, class balance: {df['quality'].mean():.1%} positive")

# ---------------------------------------------------------------------------
# 2. Ludwig config
# ---------------------------------------------------------------------------
config = {
    "model_type": "ecd",
    "input_features": [
        {"name": col, "type": "number", "preprocessing": {"normalization": "zscore"}}
        for col in df.columns
        if col != "quality"
    ],
    "output_features": [
        {"name": "quality", "type": "binary"},
    ],
    "trainer": {
        "epochs": 20,
    },
    # NOTE: Optuna executor is available from Ludwig >= 0.14 (PR #4090).
    "hyperopt": {
        "executor": {
            "type": "optuna",
            "num_samples": 20,
            # 'auto' lets Optuna pick the best sampler given the search space.
            # Alternatives: 'tpe', 'gp', 'cmaes', 'random'
            "sampler": "auto",
            # Hyperband pruner stops unpromising trials early.
            "pruner": "hyperband",
            # SQLite storage makes runs resumable: re-run the script and
            # Optuna will continue from where it left off.
            "storage": "sqlite:///optuna_results.db",
        },
        "parameters": {
            "trainer.learning_rate": {
                "space": "loguniform",
                "lower": 1e-5,
                "upper": 1e-2,
            },
            "trainer.batch_size": {
                "space": "int",
                "lower": 16,
                "upper": 256,
            },
            "trainer.optimizer.type": {
                "space": "choice",
                "categories": ["adam", "adamw", "radam", "schedule_free_adamw"],
            },
            "combiner.dropout": {
                "space": "float",
                "lower": 0.0,
                "upper": 0.5,
            },
        },
        "goal": "minimize",
        "metric": "validation.combined.loss",
        "split": "validation",
    },
}

# ---------------------------------------------------------------------------
# 3. Run HPO
# ---------------------------------------------------------------------------
try:
    from ludwig.api import LudwigModel
except ImportError:
    raise SystemExit("Ludwig is not installed. Run: pip install ludwig optuna")

print("\nStarting hyperparameter optimisation with Optuna …")
print("Results are persisted in optuna_results.db — re-run to resume.\n")

model = LudwigModel(config=config, logging_level=20)  # INFO

hyperopt_results, _, _ = model.hyperopt(
    dataset=str(combined_path),
    output_directory="hyperopt_output",
)

# ---------------------------------------------------------------------------
# 4. Report results
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("HPO complete")
print("=" * 60)

if hyperopt_results:
    best = min(hyperopt_results, key=lambda r: r.get("metric_score", float("inf")))
    print(f"\nBest metric (validation.combined.loss): {best.get('metric_score', 'n/a'):.4f}")
    print("\nBest hyperparameters:")
    for param, value in best.get("parameters", {}).items():
        print(f"  {param}: {value}")
else:
    print("No results returned — check logs above for errors.")
