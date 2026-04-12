"""Compare pretrained image encoders: stacked_cnn, DINOv2, CLIP, and SigLIP.

Runs all encoder configs on a dataset and prints a comparison table of
accuracy, training time, and approximate GPU memory usage.

Usage:
    python compare_encoders.py --dataset /path/to/data.csv

The dataset CSV must have columns: image_path, label

Example (using the beans dataset — see the notebook for download instructions):
    python compare_encoders.py --dataset /tmp/beans/train.csv \
        --val_dataset /tmp/beans/validation.csv \
        --test_dataset /tmp/beans/test.csv
"""

import argparse
import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB, or 0 if no GPU available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except ImportError:
        pass
    return 0.0


def reset_gpu_memory_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def run_experiment(
    config_path: str,
    dataset: str,
    val_dataset: str | None,
    test_dataset: str | None,
    output_dir: str,
) -> dict:
    """Train and evaluate one config.

    Returns a dict with result metrics.
    """
    from ludwig.api import LudwigModel

    reset_gpu_memory_stats()
    start = time.time()

    model = LudwigModel(config=config_path, logging_level=30)  # WARNING level

    train_kwargs = dict(dataset=dataset, output_directory=output_dir, skip_save_processed_input=True)
    if val_dataset:
        train_kwargs["validation_set"] = val_dataset

    _, _, output_directory = model.train(**train_kwargs)
    train_time = time.time() - start
    peak_mem = get_gpu_memory_mb()

    # Evaluate on test set
    eval_dataset = test_dataset or dataset
    eval_stats, _, _ = model.evaluate(dataset=eval_dataset, collect_overall_stats=True)

    accuracy = 0.0
    if "label" in eval_stats:
        accuracy = eval_stats["label"].get("accuracy", 0.0)

    return {
        "train_time_s": train_time,
        "peak_gpu_mb": peak_mem,
        "accuracy": accuracy,
        "output_directory": output_directory,
    }


CONFIGS = [
    ("stacked_cnn", "config_stacked_cnn.yaml"),
    ("dinov2_linear_probe", "config_dinov2_linear_probe.yaml"),
    ("dinov2_finetuned", "config_dinov2_finetuned.yaml"),
    ("clip", "config_clip.yaml"),
    ("siglip", "config_siglip.yaml"),
]


def print_table(results: list[dict]) -> None:
    header = f"{'Encoder':<25} {'Accuracy':>10} {'Train time':>12} {'Peak GPU (MB)':>15}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        name = r["name"]
        acc = r.get("accuracy", float("nan"))
        t = r.get("train_time_s", float("nan"))
        mem = r.get("peak_gpu_mb", 0.0)
        mem_str = f"{mem:>15.0f}" if mem > 0 else f"{'N/A':>15}"
        print(f"{name:<25} {acc:>10.4f} {t:>11.1f}s {mem_str}")
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Ludwig image encoder configs.")
    parser.add_argument("--dataset", required=True, help="Path to training CSV (image_path, label columns).")
    parser.add_argument("--val_dataset", default=None, help="Path to validation CSV.")
    parser.add_argument("--test_dataset", default=None, help="Path to test CSV (used for evaluation).")
    parser.add_argument(
        "--output_dir",
        default="/tmp/image_encoder_results",
        help="Base output directory for Ludwig experiment results.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=None,
        help="Subset of encoders to run (e.g. --encoders stacked_cnn dinov2_linear_probe). "
        "Defaults to all encoders.",
    )
    args = parser.parse_args()

    configs_to_run = CONFIGS
    if args.encoders:
        valid = {name for name, _ in CONFIGS}
        for e in args.encoders:
            if e not in valid:
                parser.error(f"Unknown encoder '{e}'. Valid: {sorted(valid)}")
        configs_to_run = [(name, cfg) for name, cfg in CONFIGS if name in args.encoders]

    results = []
    for name, config_file in configs_to_run:
        config_path = str(SCRIPT_DIR / config_file)
        if not os.path.exists(config_path):
            print(f"[SKIP] Config not found: {config_path}")
            continue

        output_dir = os.path.join(args.output_dir, name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"Config:  {config_path}")
        print(f"{'=' * 60}")

        try:
            metrics = run_experiment(
                config_path=config_path,
                dataset=args.dataset,
                val_dataset=args.val_dataset,
                test_dataset=args.test_dataset,
                output_dir=output_dir,
            )
            results.append({"name": name, **metrics})
            print(
                f"Done: accuracy={metrics['accuracy']:.4f}, "
                f"time={metrics['train_time_s']:.1f}s, "
                f"peak_gpu={metrics['peak_gpu_mb']:.0f}MB"
            )
        except Exception as exc:
            print(f"[ERROR] {name} failed: {exc}")
            results.append({"name": name, "error": str(exc)})

    print_table([r for r in results if "error" not in r])

    failed = [r for r in results if "error" in r]
    if failed:
        print("Failed experiments:")
        for r in failed:
            print(f"  {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
