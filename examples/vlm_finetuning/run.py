"""Vision-Language Model fine-tuning with Ludwig.

Fine-tunes Qwen2-VL-7B on a visual-question-answering dataset.  The dataset is
expected to be a CSV with three columns:

    image_path   — relative or absolute path to the image file
    question     — the question to ask about the image
    answer       — the expected answer (target for fine-tuning)

Usage:
    python run.py --dataset /path/to/vqa.csv --output_dir ./results
"""

import argparse
import os
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to CSV with columns: image_path, question, answer")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "vlm_config.yaml"))
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base_model in config (e.g. llava-hf/llava-1.5-7b-hf)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    from ludwig.api import LudwigModel

    overrides = {}
    if args.base_model:
        overrides["base_model"] = args.base_model

    model = LudwigModel(config=args.config, logging_level=20)  # INFO

    train_stats, _, output_dir = model.train(
        dataset=args.dataset,
        output_directory=args.output_dir,
        skip_save_processed_input=True,
    )

    print(f"\nFine-tuning complete.  Model saved to: {output_dir}")
    print("\nValidation metrics:")
    for split, metrics in train_stats.items():
        if metrics:
            print(f"  {split}: {metrics}")

    # Quick inference test
    print("\nRunning a quick inference test …")
    test_row = {
        "image_path": "test_image.jpg",
        "question": "What is in this image?",
    }
    try:
        preds, _ = model.predict(dataset=[test_row])
        print("  answer:", preds["answer_predictions"].iloc[0])
    except Exception as exc:
        print(f"  (inference test skipped — {exc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
