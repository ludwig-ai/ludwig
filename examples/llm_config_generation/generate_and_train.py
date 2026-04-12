#!/usr/bin/env python3
"""LLM-Driven Config Generation — standalone script.

Describe a machine learning task in plain English, receive a validated Ludwig
config, then optionally train a model on synthetic data.

NOTE: Requires PR #4092 to be merged into Ludwig (or ludwig>=0.14).
      https://github.com/ludwig-ai/ludwig/pull/4092

Usage
-----
# Use the built-in default task
python generate_and_train.py

# Describe your own task
python generate_and_train.py "predict house price from bedrooms, sqft, location"

# Use a specific LLM backend
python generate_and_train.py --model gpt-4o "predict house price from bedrooms, sqft"
python generate_and_train.py --model claude-sonnet-4-20250514 "predict house price"

# Colab: !pip install ludwig anthropic --quiet
"""

import argparse
import sys
import textwrap

import numpy as np
import pandas as pd
import yaml

# NOTE: `ludwig.config_generation` is provided by PR #4092 (ludwig>=0.14).
# If you see an ImportError here, that PR has not yet been merged.
try:
    from ludwig.config_generation import generate_config
except ImportError as exc:
    print(
        "ImportError: ludwig.config_generation not found.\n"
        "This feature requires PR #4092 to be merged, or ludwig>=0.14.\n"
        f"Original error: {exc}"
    )
    sys.exit(1)

DEFAULT_DESCRIPTION = (
    "I have customer data with the columns age (integer), annual_income (float), "
    "num_purchases (integer), and days_since_last_purchase (integer). "
    "I want to predict churn (binary: 0 or 1)."
)

DEFAULT_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_tabular_row(feature: dict) -> pd.Series:
    """Return a single synthetic value for a Ludwig input feature dict."""
    ftype = feature.get("type", "number")

    rng = np.random.default_rng()

    if ftype in ("number", "numerical"):
        return rng.uniform(0, 100)
    if ftype in ("category", "categorical"):
        return rng.choice(["A", "B", "C"])
    if ftype == "binary":
        return rng.choice([True, False])
    if ftype == "text":
        words = ["quick", "brown", "fox", "lazy", "dog", "jumps", "over"]
        return " ".join(rng.choice(words, size=rng.integers(4, 12)))
    # fallback
    return rng.uniform(0, 100)


def _output_value(feature: dict):
    """Return a synthetic target value for a Ludwig output feature dict."""
    ftype = feature.get("type", "binary")
    rng = np.random.default_rng()
    if ftype == "binary":
        return rng.choice([0, 1])
    if ftype in ("number", "numerical"):
        return float(rng.uniform(0, 1000))
    if ftype in ("category", "categorical"):
        return rng.choice(["cat_A", "cat_B", "cat_C"])
    return float(rng.uniform(0, 1))


def build_synthetic_dataframe(config: dict, n_rows: int = 200) -> pd.DataFrame:
    """Build a small synthetic DataFrame that matches the config schema."""
    records = []
    input_features = config.get("input_features", [])
    output_features = config.get("output_features", [])

    for _ in range(n_rows):
        row = {}
        for feat in input_features:
            row[feat["name"]] = _make_tabular_row(feat)
        for feat in output_features:
            row[feat["name"]] = _output_value(feat)
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Ludwig config from a plain-English task description and optionally train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python generate_and_train.py
              python generate_and_train.py "predict house price from bedrooms, sqft, location"
              python generate_and_train.py --model gpt-4o "classify email sentiment"
        """),
    )
    parser.add_argument(
        "description",
        nargs="?",
        default=None,
        help="Plain-English task description. Defaults to the built-in churn example.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL}). " "Claude models start with 'claude', OpenAI with 'gpt'.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Print the config but skip training.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Number of synthetic rows to generate for training (default: 200).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    description = args.description or DEFAULT_DESCRIPTION

    print("=" * 70)
    print("Ludwig LLM-Driven Config Generation")
    print("NOTE: Requires PR #4092 (ludwig>=0.14).")
    print("=" * 70)
    print()
    print("Task description:")
    print(textwrap.fill(description, width=70, initial_indent="  ", subsequent_indent="  "))
    print()
    print(f"LLM model : {args.model}")
    print()

    # ------------------------------------------------------------------
    # Generate config
    # ------------------------------------------------------------------
    print("Generating config ... (this may take a few seconds)")
    try:
        config = generate_config(
            description,
            model=args.model,
            validate=True,
        )
    except Exception as exc:
        print(f"\nError during config generation: {exc}")
        sys.exit(1)

    print("\nGenerated Ludwig config (YAML):")
    print("-" * 70)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("-" * 70)

    if args.no_train:
        print("--no-train flag set. Exiting without training.")
        return

    # ------------------------------------------------------------------
    # Confirm before training
    # ------------------------------------------------------------------
    try:
        answer = input("\nTrain a quick model on synthetic data? [y/N] ").strip().lower()
    except EOFError:
        # non-interactive environment
        answer = "n"

    if answer not in ("y", "yes"):
        print("Skipping training.")
        return

    # ------------------------------------------------------------------
    # Build synthetic dataset and train
    # ------------------------------------------------------------------
    print(f"\nBuilding synthetic dataset ({args.rows} rows) ...")
    df = build_synthetic_dataframe(config, n_rows=args.rows)
    print(f"  Columns : {list(df.columns)}")
    print(f"  Shape   : {df.shape}")

    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        csv_path = f.name

    print(f"  Saved to: {csv_path}")
    print("\nStarting Ludwig training ...")

    try:
        from ludwig.api import LudwigModel

        model = LudwigModel(config=config, logging_level="WARNING")
        train_stats, _, output_dir = model.train(
            dataset=csv_path,
            output_directory=tempfile.mkdtemp(prefix="ludwig_output_"),
        )
        print(f"\nTraining complete. Outputs saved to: {output_dir}")
        print("\nFinal validation metrics:")
        for feat_name, metrics in train_stats.get("validation", {}).items():
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {feat_name}/{metric_name}: {value:.4f}")
    except Exception as exc:
        print(f"\nTraining error: {exc}")
        print("The config generation succeeded — try adjusting the config manually.")
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    main()
