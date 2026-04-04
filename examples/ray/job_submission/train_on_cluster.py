"""Training script that runs INSIDE the Ray cluster.

This is the entrypoint for the Ray Job. It runs on the cluster head node,
so ray.init() connects locally (no Ray Client issues with ray.data).

The script expects:
  - Ludwig config at CONFIG_PATH (env var or default "config.yaml")
  - Dataset accessible from the cluster at DATASET_PATH (env var)
    This can be: S3/GCS path, NFS mount, or any path visible to the cluster.
  - Output saved to OUTPUT_DIR (env var or default "/tmp/ludwig_results")

After training, the model is saved to OUTPUT_DIR. If OUTPUT_DIR is on shared
storage (S3, GCS, NFS), it's automatically available to the submitter.
"""

import logging
import os
import sys

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # These are passed via runtime_env or set in the submission script
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    dataset_path = os.environ.get("DATASET_PATH", None)
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/ludwig_results")

    if dataset_path is None:
        logger.error("DATASET_PATH environment variable is required.")
        logger.error("Set it to an S3/GCS/NFS path accessible from the cluster.")
        sys.exit(1)

    logger.info(f"Config: {config_path}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Import Ludwig after ray.init() happens implicitly on the cluster
    from ludwig.api import LudwigModel

    # If the dataset path is a remote URI (S3, GCS), Ludwig handles it via fsspec.
    # If it's a local path, it must be accessible from the cluster node.
    model = LudwigModel(config=config, logging_level=logging.INFO)

    # Train
    # NOTE: Ludwig's Ray backend will distribute training across workers.
    # The key difference from Ray Client: this script runs ON the cluster,
    # so ray.data works correctly.
    train_stats, preprocessed_data, output_directory = model.train(
        dataset=dataset_path,
        output_directory=output_dir,
    )

    logger.info(f"Training complete. Model saved to: {output_directory}")

    # Print key metrics for the job log
    if hasattr(train_stats, "validation"):
        for feat_name, feat_metrics in train_stats.validation.items():
            if isinstance(feat_metrics, dict):
                for metric_name, values in feat_metrics.items():
                    if values and isinstance(values, list):
                        best = min(values) if "loss" in metric_name else max(values)
                        logger.info(f"  {feat_name}/{metric_name}: {best:.6f}")

    return output_directory


if __name__ == "__main__":
    main()
