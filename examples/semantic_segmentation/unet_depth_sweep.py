"""UNet depth sweep for semantic segmentation on CamSeq01.

Trains UNet models with num_stages in [2, 3, 4, 5] and reports mIoU,
approximate parameter count, and training time so you can pick the
right capacity/speed trade-off for your use case.

Usage:
    python unet_depth_sweep.py

Requirements: ludwig[vision], a CUDA-capable GPU is strongly recommended.
"""

import logging
import time

import pandas as pd
import torch
import yaml

from ludwig.api import LudwigModel
from ludwig.datasets import camseq

logging.basicConfig(level=logging.WARNING)

# ── base config ───────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "input_features": [
        {
            "name": "image_path",
            "type": "image",
            "preprocessing": {"num_processes": 4, "height": 512, "width": 512},
            "encoder": {"type": "unet"},
        }
    ],
    "output_features": [
        {
            "name": "mask_path",
            "type": "image",
            "preprocessing": {
                "num_processes": 4,
                "height": 512,
                "width": 512,
                "num_classes": 32,
            },
            "decoder": {"type": "unet", "num_fc_layers": 0, "conv_norm": "batch"},
            "loss": {"type": "softmax_cross_entropy"},
        }
    ],
    "combiner": {"type": "concat", "num_fc_layers": 0},
    "trainer": {
        "epochs": 30,
        "early_stop": 5,
        "batch_size": 4,
        "max_batch_size": 4,
        "learning_rate": 0.0001,
    },
}

DEPTHS = [2, 3, 4, 5]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_sweep():
    df = camseq.load(split=False)
    train_set = df[1:]

    results = []

    for depth in DEPTHS:
        print(f"\n{"=" * 60}")
        print(f"  Training UNet with num_stages={depth}")
        print(f"{"=" * 60}")

        config = yaml.safe_load(yaml.dump(BASE_CONFIG))  # deep copy via yaml round-trip
        config["output_features"][0]["decoder"]["num_stages"] = depth

        model = LudwigModel(config, logging_level=logging.WARNING)

        t0 = time.time()
        train_stats, _, output_dir = model.train(
            dataset=train_set,
            experiment_name="unet_depth_sweep",
            model_name=f"unet_depth_{depth}",
            skip_save_processed_input=True,
        )
        elapsed = time.time() - t0

        # Parameter count
        n_params = count_parameters(model.model)

        # Best validation loss as proxy metric (mIoU not exposed by default)
        val_loss = None
        try:
            val_history = train_stats["validation"]["combined"]["loss"]
            val_loss = min(val_history)
        except (KeyError, TypeError):
            pass

        results.append(
            {
                "num_stages": depth,
                "trainable_params": n_params,
                "best_val_loss": round(val_loss, 4) if val_loss is not None else "n/a",
                "training_time_s": round(elapsed, 1),
            }
        )

        print(f"  num_stages={depth}  params={n_params:,}  " f"best_val_loss={val_loss}  time={elapsed:.1f}s")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n\nDepth sweep summary")
    print("=" * 70)
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    print()

    return summary


if __name__ == "__main__":
    run_sweep()
