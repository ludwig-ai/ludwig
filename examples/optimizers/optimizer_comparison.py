# Colab: !pip install ludwig scikit-learn --quiet
"""
Optimizer Comparison on Wine Quality
=====================================
Compares AdamW (baseline), RAdam, Adafactor, Schedule-Free AdamW, and Muon
on a binary classification task (wine quality >= 7).

Usage:
    python optimizer_comparison.py
"""

import tempfile
import time

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/" "winequality-red.csv"

print("Downloading wine quality data...")
df = pd.read_csv(DATA_URL, sep=";")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
df["quality"] = (df["quality"] >= 7).astype(str)  # True/False binary target
print(f"Dataset shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Optimizer configs
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
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

INPUT_FEATURES = [{"name": name, "type": "number"} for name in FEATURE_NAMES]
OUTPUT_FEATURES = [{"name": "quality", "type": "binary"}]

OPTIMIZERS = {
    "adamw": {
        "trainer": {
            "epochs": 30,
            "optimizer": {"type": "adamw", "lr": 0.001},
            "learning_rate_scheduler": {"type": "cosine"},
        }
    },
    "radam": {
        "trainer": {
            "epochs": 30,
            "optimizer": {"type": "radam", "lr": 0.001},
            "learning_rate_scheduler": {"type": "cosine"},
        }
    },
    "adafactor": {
        "trainer": {
            "epochs": 30,
            "optimizer": {"type": "adafactor", "lr": 0.001},
        }
    },
    "schedule_free_adamw": {
        "trainer": {
            "epochs": 30,
            "optimizer": {"type": "schedule_free_adamw", "lr": 0.001},
            # No learning_rate_scheduler — that is the whole point of
            # Schedule-Free AdamW.
        }
    },
    "muon": {
        "trainer": {
            "epochs": 30,
            "optimizer": {"type": "muon", "lr": 0.001},
            "learning_rate_scheduler": {"type": "cosine"},
        }
    },
}

# ---------------------------------------------------------------------------
# 3. Train and collect results
# ---------------------------------------------------------------------------

from ludwig.api import LudwigModel  # noqa: E402  (import after pip install note)

results = []

for opt_name, trainer_cfg in OPTIMIZERS.items():
    print(f"\n{'=' * 60}")
    print(f"Training with optimizer: {opt_name}")
    print("=" * 60)

    config = {
        "model_type": "ecd",
        "input_features": INPUT_FEATURES,
        "output_features": OUTPUT_FEATURES,
        **trainer_cfg,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model = LudwigModel(config, logging_level=30)  # WARNING level

        t0 = time.time()
        train_stats, _, _ = model.train(
            dataset=df,
            output_directory=tmpdir,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
        )
        elapsed = time.time() - t0

    # Extract final epoch validation metrics
    val_stats = train_stats.validation
    epochs = val_stats["quality"]["loss"]
    final_loss = epochs[-1]
    final_acc = val_stats["quality"]["accuracy"][-1]

    results.append(
        {
            "optimizer": opt_name,
            "final_val_loss": round(final_loss, 4),
            "final_val_accuracy": round(final_acc, 4),
            "training_time_s": round(elapsed, 1),
        }
    )
    print(f"  val_loss={final_loss:.4f}  val_acc={final_acc:.4f}  time={elapsed:.1f}s")

# ---------------------------------------------------------------------------
# 4. Print comparison table
# ---------------------------------------------------------------------------

print("\n\nResults Summary")
print("=" * 60)
header = f"{'Optimizer':<24} {'Val Loss':>10} {'Val Acc':>10} {'Time (s)':>10}"
print(header)
print("-" * 60)
for r in results:
    print(
        f"{r['optimizer']:<24} "
        f"{r['final_val_loss']:>10.4f} "
        f"{r['final_val_accuracy']:>10.4f} "
        f"{r['training_time_s']:>10.1f}"
    )
print("=" * 60)
