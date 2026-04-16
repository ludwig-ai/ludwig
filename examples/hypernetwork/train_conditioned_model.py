"""Synthetic multi-sensor anomaly detection: HyperNetwork vs. concat combiner.

Three sensor types (A / B / C) measure the same underlying signal but with very different
scaling and noise characteristics. Anomalies are defined as readings outside a
sensor-specific envelope. A fixed-weight processing head has to find one set of
transformations that works across all three sensor profiles; the HyperNetworkCombiner
generates sensor-specific weights on the fly.

Run: ``python train_conditioned_model.py``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ludwig.api import LudwigModel

HERE = Path(__file__).parent
RNG = np.random.default_rng(42)


def make_dataset(n_per_sensor: int = 1500) -> pd.DataFrame:
    """Three sensor types, each with a different scale and anomaly envelope."""
    # (scale, noise, anomaly-threshold-on-signal)
    sensor_profiles = {
        "A": (1.0, 0.10, 1.0),
        "B": (5.0, 0.50, 4.5),
        "C": (0.2, 0.02, 0.18),
    }

    rows = []
    for sensor, (scale, noise, threshold) in sensor_profiles.items():
        signal = RNG.uniform(0.0, 1.0, size=n_per_sensor)
        noise_1 = RNG.normal(0, noise, size=n_per_sensor)
        noise_2 = RNG.normal(0, noise, size=n_per_sensor)
        noise_3 = RNG.normal(0, noise, size=n_per_sensor)
        r1 = signal * scale + noise_1
        r2 = (1 - signal) * scale + noise_2
        r3 = signal * scale * 2 + noise_3

        # Anomalies: rare large deviations calibrated to each sensor's scale.
        is_anomaly = signal > threshold
        # Inject more anomalies by perturbing 10% of points
        inject = RNG.uniform(size=n_per_sensor) < 0.1
        r1 = np.where(inject, r1 + RNG.uniform(-2, 2, n_per_sensor) * scale, r1)
        is_anomaly = is_anomaly | inject

        for a, b, c, anomaly in zip(r1, r2, r3, is_anomaly):
            rows.append(
                {
                    "sensor_type": sensor,
                    "reading_1": float(a),
                    "reading_2": float(b),
                    "reading_3": float(c),
                    "anomaly": bool(anomaly),
                }
            )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


def train(config_path: Path, dataset: pd.DataFrame, name: str) -> dict[str, float]:
    with config_path.open() as f:
        config = yaml.safe_load(f)
    model = LudwigModel(config=config, logging_level=logging.WARNING)
    _, _, output_dir = model.train(
        dataset=dataset,
        output_directory=str(HERE / f"results_{name}"),
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_model=True,
    )
    eval_stats, _, _ = model.evaluate(dataset=dataset)
    feat_stats = eval_stats["anomaly"]
    return {"accuracy": feat_stats.get("accuracy", float("nan"))}


def per_sensor_accuracy(model: LudwigModel, dataset: pd.DataFrame) -> dict[str, float]:
    preds, _ = model.predict(dataset=dataset)
    merged = dataset.copy()
    merged["predicted"] = preds["anomaly_predictions"].values
    return {sensor: float((grp["anomaly"] == grp["predicted"]).mean()) for sensor, grp in merged.groupby("sensor_type")}


def main() -> None:
    dataset = make_dataset()
    print(
        f"Dataset: {len(dataset)} rows, "
        f"{dataset['anomaly'].mean():.1%} positive, "
        f"{dataset['sensor_type'].nunique()} sensors"
    )

    results = {}
    for name, cfg in [("concat", "config_concat.yaml"), ("hypernetwork", "config_hypernetwork.yaml")]:
        print(f"\n=== Training with {name} combiner ===")
        with (HERE / cfg).open() as f:
            config = yaml.safe_load(f)
        model = LudwigModel(config=config, logging_level=logging.WARNING)
        model.train(
            dataset=dataset,
            output_directory=str(HERE / f"results_{name}"),
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            skip_save_model=True,
        )
        eval_stats, _, _ = model.evaluate(dataset=dataset)
        overall = eval_stats["anomaly"].get("accuracy", float("nan"))
        per_sensor = per_sensor_accuracy(model, dataset)
        results[name] = {"overall": overall, **per_sensor}

    print("\nPer-sensor accuracy (higher is better):")
    header = f"{'combiner':<14}" + "".join(f"{s:<10}" for s in ["overall", "A", "B", "C"])
    print(header)
    for combiner, scores in results.items():
        row = f"{combiner:<14}" + "".join(f"{scores.get(k, float('nan')):<10.4f}" for k in ["overall", "A", "B", "C"])
        print(row)

    pd.DataFrame(results).T.to_csv(HERE / "hypernetwork_results.csv")
    print(f"\nWrote {HERE / 'hypernetwork_results.csv'}")


if __name__ == "__main__":
    main()
