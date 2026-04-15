#!/usr/bin/env python
"""
Uncertainty Quantification with Ludwig: MC Dropout and Temperature Scaling.

Trains three models on the UCI Wine Quality dataset and compares:
  1. Baseline — no calibration
  2. Temperature Scaling — post-hoc calibration via a learned temperature scalar
  3. MC Dropout — per-sample uncertainty estimates via stochastic inference

Usage:
    python train.py
"""

import logging
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ludwig.api import LudwigModel

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
    """Download and prepare the wine quality dataset."""
    print("Downloading wine quality dataset...")
    df = pd.read_csv(WINE_URL, sep=";")
    # Rename columns: replace spaces with underscores
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # Binarise: quality >= 7 is "good" (True), otherwise "bad" (False)
    df["quality"] = (df["quality"] >= 7).astype(int)
    print(f"  {len(df)} rows | positive class (quality>=7): {df['quality'].mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# Ludwig configs
# ---------------------------------------------------------------------------


def _input_features() -> list:
    return [{"name": feat, "type": "number", "preprocessing": {"normalization": "zscore"}} for feat in WINE_FEATURES]


BASE_CONFIG = {
    "model_type": "ecd",
    "input_features": _input_features(),
    "output_features": [
        {
            "name": "quality",
            "type": "binary",
            "decoder": {
                "type": "mlp_classifier",
                "num_fc_layers": 1,
                "output_size": 64,
                "dropout": 0.1,
            },
            "loss": {"type": "binary_weighted_cross_entropy"},
        }
    ],
    "combiner": {
        "type": "concat",
        "num_fc_layers": 2,
        "output_size": 128,
        "dropout": 0.1,
    },
    "trainer": {"epochs": 30, "learning_rate": 0.001, "batch_size": 128},
}

CALIBRATED_CONFIG = {
    **BASE_CONFIG,
    "output_features": [
        {
            **BASE_CONFIG["output_features"][0],
            "decoder": {
                **BASE_CONFIG["output_features"][0]["decoder"],
                "calibration": "temperature_scaling",
            },
        }
    ],
}

MC_DROPOUT_CONFIG = {
    **BASE_CONFIG,
    "output_features": [
        {
            **BASE_CONFIG["output_features"][0],
            "decoder": {
                **BASE_CONFIG["output_features"][0]["decoder"],
                "dropout": 0.3,
                "mc_dropout_samples": 20,
            },
        }
    ],
    "combiner": {
        **BASE_CONFIG["combiner"],
        "dropout": 0.2,
    },
}


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        probabilities: Predicted probability for the positive class, shape (N,).
        labels: Ground-truth binary labels, shape (N,).
        n_bins: Number of equally-spaced confidence bins.

    Returns:
        ECE as a float in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probabilities)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        if mask.sum() == 0:
            continue
        conf = probabilities[mask].mean()
        acc = labels[mask].mean()
        ece += mask.sum() / n * abs(conf - acc)
    return float(ece)


def reliability_diagram(
    probabilities_dict: dict,
    labels: np.ndarray,
    n_bins: int = 10,
    output_path: str | None = None,
) -> None:
    """Plot reliability diagrams for one or more models.

    Args:
        probabilities_dict: Mapping from model name to probability arrays.
        labels: Ground-truth binary labels.
        n_bins: Number of confidence bins.
        output_path: If provided, save the figure to this path.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    colors = ["tab:red", "tab:blue", "tab:green"]
    for (name, probs), color in zip(probabilities_dict.items(), colors):
        accs = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                accs.append(float("nan"))
            else:
                accs.append(labels[mask].mean())
        ece = expected_calibration_error(probs, labels, n_bins)
        ax.plot(
            bin_centers,
            accs,
            marker="o",
            label=f"{name} (ECE={ece:.3f})",
            color=color,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Saved reliability diagram to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def uncertainty_histogram(
    uncertainty: np.ndarray,
    output_path: str | None = None,
) -> None:
    """Plot distribution of MC Dropout uncertainty estimates."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(uncertainty, bins=40, edgecolor="white", color="tab:green")
    ax.set_xlabel("Uncertainty (variance across MC samples)")
    ax.set_ylabel("Count")
    ax.set_title("MC Dropout Uncertainty Distribution")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"  Saved uncertainty histogram to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_and_evaluate(
    name: str,
    config: dict,
    dataset: pd.DataFrame,
    output_dir: str,
) -> tuple[LudwigModel, pd.DataFrame, np.ndarray]:
    """Train a Ludwig model and return predictions on the test split.

    Returns:
        (model, predictions_df, labels)
    """
    result_dir = os.path.join(output_dir, name)
    shutil.rmtree(result_dir, ignore_errors=True)

    print(f"\n--- Training: {name} ---")
    model = LudwigModel(config=config, logging_level=logging.WARNING)
    model.train(
        dataset=dataset,
        experiment_name="uncertainty",
        model_name=name,
        output_directory=result_dir,
    )

    print(f"  Evaluating {name}...")
    _, predictions, _ = model.predict(dataset=dataset)
    return model, predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_dir = "./results"
    viz_dir = "./visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Load dataset
    df = load_dataset()
    labels = df["quality"].values.astype(int)

    # Train models
    _, baseline_preds = train_and_evaluate("baseline", BASE_CONFIG, df, output_dir)
    _, calibrated_preds = train_and_evaluate("calibrated", CALIBRATED_CONFIG, df, output_dir)
    mc_model, mc_preds = train_and_evaluate("mc_dropout", MC_DROPOUT_CONFIG, df, output_dir)

    # Extract probabilities
    baseline_probs = baseline_preds["quality_probability_True"].values
    calibrated_probs = calibrated_preds["quality_probability_True"].values
    mc_probs = mc_preds["quality_probability_True"].values

    # Compute ECE
    baseline_ece = expected_calibration_error(baseline_probs, labels)
    calibrated_ece = expected_calibration_error(calibrated_probs, labels)
    mc_ece = expected_calibration_error(mc_probs, labels)

    print("\n=== Expected Calibration Error (ECE) ===")
    print(f"  Baseline:            ECE = {baseline_ece:.4f}")
    print(f"  Temperature Scaling: ECE = {calibrated_ece:.4f}")
    print(f"  MC Dropout:          ECE = {mc_ece:.4f}")
    print()

    # Reliability diagram
    reliability_diagram(
        {
            "Baseline": baseline_probs,
            "Temperature Scaling": calibrated_probs,
        },
        labels,
        output_path=os.path.join(viz_dir, "reliability_diagram.png"),
    )

    # MC Dropout uncertainty
    if "quality_uncertainty" in mc_preds.columns:
        uncertainty = mc_preds["quality_uncertainty"].values
        print("MC Dropout uncertainty stats:")
        print(f"  mean={uncertainty.mean():.4f}, std={uncertainty.std():.4f}, max={uncertainty.max():.4f}")

        uncertainty_histogram(
            uncertainty,
            output_path=os.path.join(viz_dir, "mc_dropout_uncertainty.png"),
        )

        # Show high-uncertainty predictions
        threshold = np.percentile(uncertainty, 80)
        high_unc_mask = uncertainty >= threshold
        print(f"\nHigh-uncertainty samples (top 20%, threshold={threshold:.4f}):")
        high_unc_preds = mc_preds["quality_predictions"].values[high_unc_mask].astype(bool)
        high_unc_labels = labels[high_unc_mask].astype(bool)
        high_unc_acc = (high_unc_preds == high_unc_labels).mean()
        print(f"  count={high_unc_mask.sum()}, accuracy on these samples: {high_unc_acc:.2%}")
    else:
        print("Note: 'quality_uncertainty' column not found in predictions.")
        print("Make sure mc_dropout_samples > 0 and the decoder has dropout > 0.")

    print(f"\nDone. Plots saved to {viz_dir}/")


if __name__ == "__main__":
    main()
