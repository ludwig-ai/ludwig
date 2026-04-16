"""End-to-end demo: natural-language task description -> Ludwig config -> trained model.

Reads a task description from ``example_description.txt`` (or stdin), calls
``ludwig.config_generation.generate_config`` to turn it into a validated Ludwig config,
writes the config to disk, and trains on the UCI Adult Census Income dataset.

Requires Ludwig 0.15 / PR #4092 and an ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` env var.

Run: ``python generate_and_train.py``
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent


def load_description() -> str:
    desc_file = HERE / "example_description.txt"
    if desc_file.exists():
        return desc_file.read_text().strip()
    print("example_description.txt not found. Paste a task description then Ctrl+D:", file=sys.stderr)
    return sys.stdin.read().strip()


def main() -> None:
    from ludwig.api import LudwigModel
    from ludwig.config_generation import generate_config
    from ludwig.datasets import adult_census_income

    if "ANTHROPIC_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment before running this script.")

    description = load_description()
    print("Generating config from task description…\n")

    # Auto-select model based on which API key is present.
    if "ANTHROPIC_API_KEY" in os.environ:
        model = "claude-sonnet-4-20250514"
    else:
        model = "gpt-4o"

    config = generate_config(
        task_description=description,
        model=model,
        validate=True,  # raises if the LLM produces an invalid config
    )

    config_path = HERE / "generated_config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote validated config to {config_path}\n")

    print("Loading Adult Census Income dataset…")
    dataset = adult_census_income.load()

    print("Training…")
    ludwig_model = LudwigModel(config=config, logging_level=logging.INFO)
    train_stats, _, output_dir = ludwig_model.train(
        dataset=dataset,
        output_directory=str(HERE / "results"),
        skip_save_processed_input=True,
    )
    print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
