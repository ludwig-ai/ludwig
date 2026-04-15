# Colab: !pip install ludwig
"""Anomaly detection with Ludwig using Deep SVDD, Deep SAD, and DROCC losses.

Generates synthetic sensor data, trains all three model variants, evaluates on a
held-out test set containing both normal and anomalous samples, and prints an AUC-ROC
comparison table.

Usage:
    python train.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ludwig.api import LudwigModel

# ---------------------------------------------------------------------------
# 1. Generate synthetic sensor data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_NORMAL = 800
N_ANOMALY = 200

# Normal samples: Gaussian near origin
normal = pd.DataFrame(
    {
        "sensor_a": RNG.normal(0.0, 1.0, N_NORMAL),
        "sensor_b": RNG.normal(0.0, 1.0, N_NORMAL),
        "sensor_c": RNG.normal(0.0, 1.0, N_NORMAL),
        "timestamp_hour": RNG.integers(0, 24, N_NORMAL).astype(float),
        "anomaly": 0.0,
    }
)

# Anomalous samples: large offset from origin
anomalous = pd.DataFrame(
    {
        "sensor_a": RNG.normal(6.0, 1.0, N_ANOMALY),
        "sensor_b": RNG.normal(6.0, 1.0, N_ANOMALY),
        "sensor_c": RNG.normal(6.0, 1.0, N_ANOMALY),
        "timestamp_hour": RNG.integers(0, 24, N_ANOMALY).astype(float),
        "anomaly": 1.0,
    }
)

all_data = pd.concat([normal, anomalous], ignore_index=True)

# Train split: ONLY normal samples (anomaly detection is unsupervised)
# Val split: mix of normal and anomalous for threshold selection
# Test split: mix for final evaluation

normal_idx = all_data[all_data["anomaly"] == 0].index.tolist()
anomaly_idx = all_data[all_data["anomaly"] == 1].index.tolist()

RNG.shuffle(normal_idx)
n_train = int(0.7 * len(normal_idx))
n_val = int(0.15 * len(normal_idx))

train_idx = normal_idx[:n_train]
val_normal_idx = normal_idx[n_train : n_train + n_val]
test_normal_idx = normal_idx[n_train + n_val :]

RNG.shuffle(anomaly_idx)
n_val_anom = len(anomaly_idx) // 2
val_anom_idx = anomaly_idx[:n_val_anom]
test_anom_idx = anomaly_idx[n_val_anom:]

# Assign split column: 0=train, 1=val, 2=test
all_data["split"] = -1
all_data.loc[train_idx, "split"] = 0
all_data.loc[val_normal_idx, "split"] = 1
all_data.loc[val_anom_idx, "split"] = 1
all_data.loc[test_normal_idx, "split"] = 2
all_data.loc[test_anom_idx, "split"] = 2

train_df = all_data[all_data["split"] == 0].copy()
test_df = all_data[all_data["split"].isin([1, 2])].copy()

train_df.to_csv("/tmp/sensors_train.csv", index=False)
test_df.to_csv("/tmp/sensors_test.csv", index=False)

print(f"Train samples: {len(train_df)}  (all normal)")
print(f"Test  samples: {len(test_df)}  ({(test_df['anomaly'] == 1).sum()} anomalous)")

# ---------------------------------------------------------------------------
# 2. Helper: build Ludwig config dict
# ---------------------------------------------------------------------------

INPUT_FEATURES = [
    {"name": "sensor_a", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "sensor_b", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "sensor_c", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "timestamp_hour", "type": "number", "preprocessing": {"normalization": "zscore"}},
]

COMBINER = {
    "type": "concat",
    "fc_layers": [{"output_size": 64}, {"output_size": 32}],
}

TRAINER = {"epochs": 20, "learning_rate": 0.001}


def make_config(loss: dict) -> dict:
    return {
        "model_type": "ecd",
        "input_features": INPUT_FEATURES,
        "output_features": [
            {
                "name": "anomaly",
                "type": "anomaly",
                "loss": loss,
            }
        ],
        "combiner": COMBINER,
        "trainer": TRAINER,
    }


CONFIGS = {
    "Deep SVDD": make_config({"type": "deep_svdd", "nu": 0.1}),
    "Deep SAD": make_config({"type": "deep_sad", "eta": 1.0}),
    "DROCC": make_config({"type": "drocc", "perturbation_strength": 0.1, "num_perturbation_steps": 5}),
}

# Deep SAD: inject ~10% labeled anomalies into the training set
N_LABELED = max(1, int(0.1 * len(train_df)))
labeled_anom = anomalous.sample(n=N_LABELED, random_state=0).copy()
labeled_anom["split"] = 0
sad_train_df = pd.concat([train_df, labeled_anom], ignore_index=True)

# ---------------------------------------------------------------------------
# 3. Train and evaluate each variant
# ---------------------------------------------------------------------------

results_table = []

for method_name, config in CONFIGS.items():
    print(f"\n{'=' * 60}")
    print(f"Training: {method_name}")
    print("=" * 60)

    train_data = sad_train_df if method_name == "Deep SAD" else train_df

    model = LudwigModel(config, logging_level=30)  # WARNING level
    train_stats, _, _ = model.train(dataset=train_data)

    predictions, _ = model.predict(dataset=test_df)

    score_col = "anomaly_anomaly_score_predictions"
    scores = predictions[score_col].values
    true_labels = test_df["anomaly"].values

    auc = roc_auc_score(true_labels, scores)

    # Separation ratio: mean anomaly score / mean normal score
    normal_scores = scores[true_labels == 0]
    anom_scores = scores[true_labels == 1]
    sep_ratio = anom_scores.mean() / (normal_scores.mean() + 1e-9)

    results_table.append(
        {
            "Method": method_name,
            "AUC-ROC": round(auc, 4),
            "Mean normal score": round(float(normal_scores.mean()), 4),
            "Mean anomaly score": round(float(anom_scores.mean()), 4),
            "Separation ratio": round(float(sep_ratio), 2),
        }
    )
    print(f"  AUC-ROC:          {auc:.4f}")
    print(f"  Mean normal score: {normal_scores.mean():.4f}")
    print(f"  Mean anomaly score:{anom_scores.mean():.4f}")
    print(f"  Separation ratio:  {sep_ratio:.2f}x")

# ---------------------------------------------------------------------------
# 4. Print summary table
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results_table)

print("\n")
print("=" * 70)
print("ANOMALY DETECTION — SUMMARY")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)
print("\nHigher AUC-ROC and separation ratio indicate better discrimination " "between normal and anomalous samples.")
