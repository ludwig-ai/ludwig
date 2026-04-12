# Colab: !pip install "ludwig>=0.14"
"""HyperNetworkCombiner vs concat — sensor anomaly detection.

Generates a synthetic multi-modal sensor dataset where the correct interpretation
of numerical sensor readings depends entirely on the sensor type (temperature,
pressure, or humidity). Trains a baseline concat model and a HyperNetworkCombiner
model, then prints an accuracy comparison.

NOTE: Requires ludwig >= 0.14 (PR #4092). The hypernetwork combiner is not
available in earlier versions.

Usage:
    python train_hypernetwork.py
"""

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel

# ---------------------------------------------------------------------------
# 1. Generate synthetic sensor data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_PER_TYPE = 600
SENSOR_TYPES = ["temperature", "pressure", "humidity"]


def make_samples(sensor_type: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate n samples for a single sensor type.

    Each type has its own 'normal' operating range and anomaly rule so that the same raw reading can mean very different
    things depending on the type.
    """
    if sensor_type == "temperature":
        # Normal: sensors cluster near (0, 0, 0); anomaly: sensor_a > 2.5
        sensor_a = rng.normal(0.0, 1.0, n)
        sensor_b = rng.normal(0.0, 1.0, n)
        sensor_c = rng.normal(0.0, 1.0, n)
        anomaly = (sensor_a > 2.5).astype(int)
    elif sensor_type == "pressure":
        # Normal: sensors cluster near (1, 1, 1); anomaly: sensor_b < -1.5
        sensor_a = rng.normal(1.0, 0.8, n)
        sensor_b = rng.normal(1.0, 0.8, n)
        sensor_c = rng.normal(1.0, 0.8, n)
        anomaly = (sensor_b < -0.5).astype(int)
    else:  # humidity
        # Normal: sensors cluster near (-1, -1, -1); anomaly: sum > 0
        sensor_a = rng.normal(-1.0, 0.9, n)
        sensor_b = rng.normal(-1.0, 0.9, n)
        sensor_c = rng.normal(-1.0, 0.9, n)
        anomaly = ((sensor_a + sensor_b + sensor_c) > 0).astype(int)

    return pd.DataFrame(
        {
            "sensor_a": sensor_a,
            "sensor_b": sensor_b,
            "sensor_c": sensor_c,
            "sensor_type": sensor_type,
            "anomaly": anomaly,
        }
    )


frames = [make_samples(t, N_PER_TYPE, RNG) for t in SENSOR_TYPES]
df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Train / validation / test split  (70 / 15 / 15)
n = len(df)
split = np.full(n, 2, dtype=int)  # default: test
idx = np.arange(n)
RNG.shuffle(idx)
split[idx[: int(0.70 * n)]] = 0
split[idx[int(0.70 * n) : int(0.85 * n)]] = 1
df["split"] = split

print(f"Dataset: {n} rows  ({df['anomaly'].mean():.1%} anomalies)")
print(df.groupby("sensor_type")["anomaly"].mean().rename("anomaly_rate").to_string())
print()

# ---------------------------------------------------------------------------
# 2. Model configs
# ---------------------------------------------------------------------------

INPUT_FEATURES = [
    {"name": "sensor_a", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "sensor_b", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "sensor_c", "type": "number", "preprocessing": {"normalization": "zscore"}},
    {"name": "sensor_type", "type": "category"},
]

OUTPUT_FEATURES = [{"name": "anomaly", "type": "binary"}]

TRAINER = {"epochs": 30, "learning_rate": 0.001}

config_concat = {
    "model_type": "ecd",
    "input_features": INPUT_FEATURES,
    "output_features": OUTPUT_FEATURES,
    "combiner": {
        "type": "concat",
        "fc_layers": [{"output_size": 128}, {"output_size": 64}],
    },
    "trainer": TRAINER,
}

config_hypernetwork = {
    "model_type": "ecd",
    "input_features": INPUT_FEATURES,
    "output_features": OUTPUT_FEATURES,
    "combiner": {
        "type": "hypernetwork",
        "hidden_size": 128,
        "hyper_hidden_size": 64,
        "output_size": 128,
    },
    "trainer": TRAINER,
}

# ---------------------------------------------------------------------------
# 3. Train and evaluate
# ---------------------------------------------------------------------------

results = []

for label, config in [("Concat (baseline)", config_concat), ("HyperNetwork", config_hypernetwork)]:
    print(f"{'=' * 60}")
    print(f"Training: {label}")
    print("=" * 60)

    model = LudwigModel(config, logging_level=30)
    model.train(dataset=df)

    test_df = df[df["split"] == 2].copy()
    predictions, _ = model.predict(dataset=test_df)

    pred_col = "anomaly_predictions"
    correct = (predictions[pred_col].values == test_df["anomaly"].values).mean()

    results.append({"Model": label, "Test accuracy": round(float(correct), 4)})
    print(f"  Test accuracy: {correct:.4f}\n")

# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("=" * 50)
print("SENSOR ANOMALY DETECTION — SUMMARY")
print("=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)
print()
print("The HyperNetworkCombiner lets sensor_type rewrite the")
print("transformation applied to sensor_a/b/c rather than")
print("just concatenating all features together.")
