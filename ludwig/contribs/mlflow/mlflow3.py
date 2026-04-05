"""MLflow 3.x integration enhancements.

Adds support for MLflow 3.x features:
- LoggedModel: model-centric tracking (not run-centric)
- GenAI tracing: structured logging of LLM prompts, responses, and evaluation
- Cost tracking: automatic model info extraction for cost estimation

These features supplement the existing MLflow integration in model.py.
They are opt-in and gracefully degrade if MLflow 3.x is not available.

Usage:
    from ludwig.contribs.mlflow.mlflow3 import log_training_run

    # After training:
    log_training_run(
        model=ludwig_model,
        train_stats=train_stats,
        config=config,
    )
"""

import logging

logger = logging.getLogger(__name__)


def log_training_run(model, train_stats=None, config=None, tags=None):
    """Log a Ludwig training run to MLflow with 3.x features.

    Creates a LoggedModel entity (if MLflow 3.x is available) that is
    model-centric rather than run-centric. This allows tracking the
    model across multiple runs (retraining, fine-tuning, evaluation).

    Args:
        model: Trained LudwigModel instance.
        train_stats: Training statistics from model.train().
        config: Ludwig config dict.
        tags: Additional MLflow tags.
    """
    try:
        import mlflow

        mlflow_version = tuple(int(x) for x in mlflow.__version__.split(".")[:2])
    except ImportError:
        logger.warning("MLflow not installed. Skipping training run logging.")
        return

    with mlflow.start_run(nested=True) as run:
        # Log config as params
        if config:
            _log_config_params(config)

        # Log training metrics
        if train_stats:
            _log_training_metrics(train_stats)

        # Log model info and cost tracking
        import ludwig

        mlflow.set_tag("ludwig.version", ludwig.__version__)
        mlflow.set_tag("model.type", config.get("model_type", "ecd") if config else "unknown")

        # Cost tracking: log model size and parameter counts for cost estimation
        if model and hasattr(model, "model") and model.model is not None:
            try:
                total_params = sum(p.numel() for p in model.model.parameters())
                trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
                model_size_mb = sum(p.numel() * p.element_size() for p in model.model.parameters()) / (1024 * 1024)
                mlflow.log_metric("model.total_params", total_params)
                mlflow.log_metric("model.trainable_params", trainable_params)
                mlflow.log_metric("model.size_mb", round(model_size_mb, 2))
                mlflow.set_tag("model.param_efficiency", f"{trainable_params / max(total_params, 1) * 100:.1f}%")
            except Exception:
                pass

            # Log base model name for LLMs (useful for cost estimation)
            if config and config.get("model_type") == "llm":
                base_model = config.get("base_model", "unknown")
                mlflow.set_tag("model.base_model", base_model)

        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Log model artifact
        try:
            from ludwig.contribs.mlflow.model import log_model

            log_model(model, artifact_path="ludwig-model")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

        # MLflow 3.x: Create LoggedModel if available
        if mlflow_version >= (3, 0):
            try:
                _create_logged_model(run, model, config)
            except Exception as e:
                logger.debug(f"MLflow 3.x LoggedModel not available: {e}")

        logger.info(f"Training run logged to MLflow: {run.info.run_id}")
        return run.info.run_id


def log_llm_trace(prompt, response, model_name=None, latency_ms=None, tokens_used=None):
    """Log an LLM inference trace to MLflow GenAI tracking.

    MLflow 3.x provides structured tracing for LLM interactions including
    prompts, responses, latency, and token usage.

    Args:
        prompt: The input prompt text.
        response: The model's response text.
        model_name: Name/ID of the model used.
        latency_ms: Inference latency in milliseconds.
        tokens_used: Number of tokens consumed.
    """
    try:
        import mlflow

        mlflow_version = tuple(int(x) for x in mlflow.__version__.split(".")[:2])
        if mlflow_version < (3, 0):
            logger.debug("MLflow GenAI tracing requires MLflow 3.x")
            return
    except ImportError:
        return

    try:
        # MLflow 3.x GenAI tracing
        mlflow.log_table(
            data={
                "prompt": [prompt],
                "response": [response],
                "model": [model_name or "ludwig-llm"],
                "latency_ms": [latency_ms],
                "tokens": [tokens_used],
            },
            artifact_file="genai_traces.json",
        )
    except Exception as e:
        logger.debug(f"GenAI tracing failed: {e}")


def _log_config_params(config):
    """Log Ludwig config as MLflow params (flattened)."""
    import mlflow

    def flatten(d, prefix=""):
        params = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                params.update(flatten(v, key))
            elif isinstance(v, (list, tuple)):
                params[key] = str(v)[:250]  # MLflow param value limit
            else:
                params[key] = str(v)[:250]
        return params

    flat = flatten(config)
    # MLflow has a 100-param batch limit
    items = list(flat.items())
    for i in range(0, len(items), 100):
        batch = dict(items[i : i + 100])
        try:
            mlflow.log_params(batch)
        except Exception:
            pass  # Skip params that fail (duplicates, etc.)


def _log_training_metrics(train_stats):
    """Log best training metrics to MLflow."""
    import mlflow

    for split_name in ["validation", "test"]:
        split_data = getattr(train_stats, split_name, None)
        if not split_data:
            continue
        for feat_name, feat_metrics in split_data.items():
            if not isinstance(feat_metrics, dict):
                continue
            for metric_name, values in feat_metrics.items():
                if isinstance(values, list) and values:
                    if "loss" in metric_name or "error" in metric_name:
                        best = min(values)
                    else:
                        best = max(values)
                    try:
                        mlflow.log_metric(f"{split_name}.{feat_name}.{metric_name}", best)
                    except Exception:
                        pass


def _create_logged_model(run, model, config):
    """Create an MLflow 3.x LoggedModel entity."""
    import mlflow

    try:
        # MLflow 3.x API
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/ludwig-model",
            name=config.get("model_name", "ludwig-model") if config else "ludwig-model",
        )
    except Exception as e:
        logger.debug(f"Model registration failed: {e}")
