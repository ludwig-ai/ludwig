"""Side-by-side optimizer comparison on the Wine Quality regression benchmark.

Trains the same tabular model with each of AdamW / RAdam / Schedule-Free AdamW / Muon
/ Adafactor and plots loss + RMSE learning curves for each. Results are written to
``optimizer_comparison.csv`` and ``optimizer_comparison.png`` in the current directory.

Run: ``python optimizer_comparison.py``

Optional installs for the full comparison (skipped gracefully if missing):
    pip install transformers      # Adafactor
    pip install schedulefree      # Schedule-Free AdamW
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import wine_quality

CONFIGS = [
    ("adamw", "config_adamw.yaml"),
    ("radam", "config_radam.yaml"),
    ("schedule_free_adamw", "config_schedule_free_adamw.yaml"),
    ("muon", "config_muon.yaml"),
    ("adafactor", "config_adafactor.yaml"),
]

HERE = Path(__file__).parent


def run_one(name: str, config_path: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    """Train once and return per-epoch validation metrics as a tidy frame."""
    with config_path.open() as f:
        config = yaml.safe_load(f)

    model = LudwigModel(config=config, logging_level=logging.WARNING)
    train_stats, _, _ = model.train(
        dataset=dataset,
        output_directory=str(HERE / f"results_{name}"),
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_model=True,
    )
    val_stats = train_stats.validation_stats["quality"]
    loss = val_stats.get("loss", [])
    rmse = val_stats.get("root_mean_squared_error", [])
    return pd.DataFrame(
        {
            "optimizer": name,
            "epoch": range(1, len(loss) + 1),
            "val_loss": loss,
            "val_rmse": rmse,
        }
    )


def safe_run(name: str, config_path: Path, dataset: pd.DataFrame) -> pd.DataFrame | None:
    """Run a single config, skipping if the optional dependency is missing."""
    try:
        return run_one(name, config_path, dataset)
    except ImportError as e:
        print(f"[skip] {name}: missing optional dependency ({e}). Skipping.")
        return None


def main() -> None:
    dataset = wine_quality.load()

    frames = []
    for name, fname in CONFIGS:
        print(f"\n=== Training with {name} ===")
        df = safe_run(name, HERE / fname, dataset)
        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit("No optimizer runs completed successfully.")

    out = pd.concat(frames, ignore_index=True)
    csv_path = HERE / "optimizer_comparison.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Plot if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. pip install 'ludwig[viz]' to get it.")
        return

    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, group in out.groupby("optimizer"):
        ax_loss.plot(group["epoch"], group["val_loss"], label=name)
        ax_rmse.plot(group["epoch"], group["val_rmse"], label=name)
    for ax, title in ((ax_loss, "Validation loss"), (ax_rmse, "Validation RMSE")):
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Wine Quality — optimizer comparison")
    fig.tight_layout()
    png_path = HERE / "optimizer_comparison.png"
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}")

    summary = (
        out.groupby("optimizer")
        .agg(best_rmse=("val_rmse", "min"), final_rmse=("val_rmse", "last"))
        .sort_values("best_rmse")
    )
    print("\nBest validation RMSE per optimizer:")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    (HERE / "summary.json").write_text(json.dumps(summary.to_dict(orient="index"), indent=2))


if __name__ == "__main__":
    main()
