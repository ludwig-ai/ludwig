"""Minimal training demo for advanced PEFT adapters in Ludwig.

Run with: python train_example.py --adapter pissa
"""

from __future__ import annotations

import argparse
import logging

import yaml

from ludwig.api import LudwigModel

ADAPTER_CONFIGS = {
    "lora": {"type": "lora", "r": 8, "alpha": 16},
    "pissa": {"type": "lora", "r": 8, "alpha": 16, "init_lora_weights": "pissa"},
    "corda": {"type": "lora", "r": 8, "alpha": 16, "init_lora_weights": "corda"},
    "rslora": {"type": "lora", "r": 8, "alpha": 16, "use_rslora": True},
    "dora": {"type": "lora", "r": 8, "alpha": 16, "use_dora": True},
    "tinylora": {"type": "tinylora", "r": 2, "u": 64},
    "oft": {"type": "oft", "oft_block_size": 32},
    "hra": {"type": "hra", "r": 8},
    "ln_tuning": {"type": "ln_tuning"},
    "vblora": {"type": "vblora", "r": 4, "num_vectors": 256, "vector_length": 768, "topk": 2},
    "adalora": {"type": "adalora", "r": 8, "target_r": 4, "init_r": 12, "total_step": 1000},
    "ia3": {"type": "ia3"},
    "vera": {"type": "vera", "r": 256},
}

BASE_CONFIG = """
model_type: llm
base_model: sshleifer/tiny-gpt2

input_features:
  - name: text
    type: text

output_features:
  - name: label
    type: text

trainer:
  type: finetune
  learning_rate: 0.0001
  epochs: 1
  batch_size: 4
"""


def main():
    parser = argparse.ArgumentParser(description="Train with a specific PEFT adapter")
    parser.add_argument("--adapter", default="pissa", choices=list(ADAPTER_CONFIGS.keys()))
    parser.add_argument("--dataset", default=None, help="Path to dataset CSV (optional)")
    args = parser.parse_args()

    config = yaml.safe_load(BASE_CONFIG)
    config["adapter"] = ADAPTER_CONFIGS[args.adapter]

    print(f"Training with adapter: {args.adapter}")
    print(f"Adapter config: {config['adapter']}")
    print()

    model = LudwigModel(config=config, logging_level=logging.INFO)

    if args.dataset:
        stats, _, output_dir = model.train(dataset=args.dataset)
        print(f"Training complete. Results in: {output_dir}")
    else:
        print("No dataset provided — config validated successfully.")
        print("Pass --dataset <path> to run training.")


if __name__ == "__main__":
    main()
