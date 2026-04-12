"""DPO alignment training with Ludwig.

Usage:
    python train_dpo.py --dataset train.csv
    python train_dpo.py --dataset train.csv --epochs 3 --beta 0.05

Prerequisites:
    pip install "ludwig[llm]" datasets
    export HUGGING_FACE_HUB_TOKEN="<your_token>"

The dataset must have columns: prompt, chosen, rejected
Use prepare_dataset.py to produce this file from Anthropic/hh-rlhf.
"""

import argparse
import logging
import os

import yaml

from ludwig.api import LudwigModel


def build_config(epochs: int, learning_rate: float, beta: float, batch_size: int) -> dict:
    raw = f"""
model_type: llm
base_model: meta-llama/Llama-3.1-8B

adapter:
  type: lora
  r: 16
  alpha: 32
  dropout: 0.05

trainer:
  type: dpo
  epochs: {epochs}
  learning_rate: {learning_rate}
  batch_size: {batch_size}
  gradient_accumulation_steps: 8
  beta: {beta}

input_features:
  - name: prompt
    type: text

output_features:
  - name: chosen
    type: text

backend:
  type: local
"""
    return yaml.safe_load(raw)


def main():
    parser = argparse.ArgumentParser(description="Run DPO alignment training with Ludwig.")
    parser.add_argument("--dataset", required=True, help="Path to the DPO CSV (prompt, chosen, rejected).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--experiment_name", default="hh_rlhf_dpo")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise OSError(
            "Set HUGGING_FACE_HUB_TOKEN (or HF_TOKEN) before running. "
            "You also need access approval for meta-llama/Llama-3.1-8B."
        )

    config = build_config(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        batch_size=args.batch_size,
    )

    model = LudwigModel(config=config, logging_level=logging.INFO)

    train_stats, preprocessed_data, output_directory = model.train(
        dataset=args.dataset,
        experiment_name=args.experiment_name,
        output_directory=args.output_dir,
    )

    print(f"\nTraining complete. Results saved to: {output_directory}")
    print("To upload the model to HuggingFace Hub:")
    print(f"    ludwig upload hf_hub -r <your_org>/<model_name> -m {output_directory}")


if __name__ == "__main__":
    main()
