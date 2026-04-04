"""Generate a structured training report JSON.

Captures the full provenance of a training run: config, data schema,
metrics, hardware, timing, and Ludwig version. Useful for audit trails,
compliance documentation, and reproducibility.
"""

import logging
import os
import platform
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def generate_training_report(
    config: dict,
    training_set_metadata: dict,
    train_stats=None,
    output_directory: str | None = None,
    model_dir: str | None = None,
    dataset_statistics: list | None = None,
    random_seed: int | None = None,
    training_time_seconds: float | None = None,
) -> dict:
    """Generate a structured training report.

    Args:
        config: The full Ludwig config dict.
        training_set_metadata: Feature metadata computed during preprocessing.
        train_stats: Training statistics (train/validation/test metrics per epoch).
        output_directory: Path to the experiment output directory.
        model_dir: Path where the model is saved.
        dataset_statistics: Dataset split sizes.
        random_seed: Random seed used for training.
        training_time_seconds: Total training time.

    Returns:
        Dict with full training provenance.
    """
    report = {
        "report_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Environment
    env = {"python_version": platform.python_version(), "platform": platform.platform()}
    try:
        import ludwig

        env["ludwig_version"] = ludwig.__version__
    except ImportError:
        pass
    try:
        import torch

        env["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    report["environment"] = env

    # Config
    report["config"] = config
    report["model_type"] = config.get("model_type", "ecd")
    report["random_seed"] = random_seed

    # Data schema: what features were used, their types, and key metadata
    data_schema = {"input_features": [], "output_features": []}
    for feat in config.get("input_features", []):
        feat_info = {"name": feat["name"], "type": feat["type"]}
        meta = training_set_metadata.get(feat["name"], {})
        if isinstance(meta, dict):
            if "mean" in meta:
                feat_info["mean"] = meta["mean"]
                feat_info["std"] = meta.get("std")
            if "idx2str" in meta:
                feat_info["vocab_size"] = len(meta["idx2str"])
        data_schema["input_features"].append(feat_info)

    for feat in config.get("output_features", []):
        feat_info = {"name": feat["name"], "type": feat["type"]}
        meta = training_set_metadata.get(feat["name"], {})
        if isinstance(meta, dict):
            if "idx2str" in meta:
                feat_info["vocab_size"] = len(meta["idx2str"])
                feat_info["classes"] = meta["idx2str"]
        data_schema["output_features"].append(feat_info)
    report["data_schema"] = data_schema

    # Dataset statistics
    if dataset_statistics:
        ds_stats = {}
        for row in dataset_statistics:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                ds_stats[str(row[0])] = row[1]
        report["dataset_statistics"] = ds_stats

    # Training metrics: best value per metric per feature per split
    if train_stats is not None:
        metrics = {}
        for split_name, split_attr in [("training", "training"), ("validation", "validation"), ("test", "test")]:
            split_data = getattr(train_stats, split_attr, None)
            if split_data:
                split_metrics = {}
                for feat_name, feat_metrics in split_data.items():
                    if isinstance(feat_metrics, dict):
                        feat_best = {}
                        for metric_name, values in feat_metrics.items():
                            if isinstance(values, list) and values:
                                if "loss" in metric_name or "error" in metric_name:
                                    feat_best[metric_name] = {"best": min(values), "last": values[-1]}
                                else:
                                    feat_best[metric_name] = {"best": max(values), "last": values[-1]}
                        if feat_best:
                            split_metrics[feat_name] = feat_best
                if split_metrics:
                    metrics[split_name] = split_metrics
        report["metrics"] = metrics

        # Epochs trained
        combined = getattr(train_stats, "training", {})
        if isinstance(combined, dict):
            combined_metrics = combined.get("combined", {})
            loss_values = combined_metrics.get("loss", [])
            if loss_values:
                report["epochs_trained"] = len(loss_values)

    # Timing
    if training_time_seconds is not None:
        report["training_time_seconds"] = round(training_time_seconds, 2)

    # Paths
    if output_directory:
        report["output_directory"] = output_directory
    if model_dir:
        report["model_directory"] = model_dir

    return report


def save_training_report(
    output_directory: str,
    config: dict,
    training_set_metadata: dict,
    train_stats=None,
    model_dir: str | None = None,
    dataset_statistics: list | None = None,
    random_seed: int | None = None,
    training_time_seconds: float | None = None,
):
    """Generate and save a training report JSON to the output directory."""
    from ludwig.utils.data_utils import save_json

    report = generate_training_report(
        config=config,
        training_set_metadata=training_set_metadata,
        train_stats=train_stats,
        output_directory=output_directory,
        model_dir=model_dir,
        dataset_statistics=dataset_statistics,
        random_seed=random_seed,
        training_time_seconds=training_time_seconds,
    )

    report_path = os.path.join(output_directory, "training_report.json")
    save_json(report_path, report)
    logger.info(f"Training report saved to {report_path}")
    return report_path
