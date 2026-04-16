"""Group Relative Policy Optimization (GRPO) fine-tuning example.

GRPO (Shao et al., 2024) is the alignment method that powers DeepSeek-R1 and DeepSeek-Math.
For each prompt it samples a *group* of completions, scores them with a reward function,
normalizes rewards within the group, and uses those normalized advantages in a PPO-style
clipped objective. No critic model is required.

This script uses the same `chosen` / `rejected` preference-pair format as DPO — the
implicit reward within each group is simply `+1` for the chosen completion and `-1` for
the rejected one, so GRPO reduces to a clipped preference loss with a KL penalty toward
the reference policy. Use this as a starting point and swap in a real reward function by
pre-scoring completions in your dataset preparation step (see the ``prepare_dataset.py``
companion script in this directory for the DPO/KTO pattern).

Run: ``python train_grpo.py``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from ludwig.api import LudwigModel

HERE = Path(__file__).parent


def main() -> None:
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        print(
            "Tip: set HUGGING_FACE_HUB_TOKEN if the chosen base_model requires authentication.\n"
            "The default SmolLM2 model is public and does not."
        )

    with (HERE / "config_grpo.yaml").open() as f:
        config = yaml.safe_load(f)

    # Reuse the preference dataset emitted by prepare_dataset.py. If not present,
    # prompt the user to generate it first.
    dataset_path = HERE / "preference_data.parquet"
    if not dataset_path.exists():
        raise SystemExit(
            f"{dataset_path} not found. Run `python prepare_dataset.py` first to emit a\n"
            "preference-pair parquet file with `prompt`, `output` (chosen), and `rejected` columns."
        )

    model = LudwigModel(config=config, logging_level=logging.INFO)
    model.train(
        dataset=str(dataset_path),
        output_directory=str(HERE / "results_grpo"),
        skip_save_processed_input=True,
    )


if __name__ == "__main__":
    main()
