import os
from typing import Union

from ludwig.schema.model_types.base import ModelConfig


def _default_transformers_cache_dir() -> str:
    """Return the default cache directory for transformers.

    This function returns the default cache directory used by the transformers library for storing downloaded
    models and data.

    Returns:
        str: The default cache directory path.
    """
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache", "huggingface", "hub")
    return cache_dir


def _get_backend_type_from_config(config_obj: ModelConfig) -> str:
    """Get the backend type from a model configuration object.

    This function retrieves the backend type specified in a language model configuration.

    Args:
        config_obj (ModelConfig): The configuration object containing backend information.

    Returns:
        str: The backend type, defaulting to "local" if not explicitly specified.
    """
    backend = config_obj.backend
    backend_type = backend.get("type", "local")
    return backend_type


def _get_deepspeed_optimization_stage_from_config(config_obj: ModelConfig) -> Union[int, None]:
    """Get the DeepSpeed optimization stage from a model configuration object.

    This function extracts the DeepSpeed optimization stage from a language model configuration if the backend type
    is set to "ray" and the strategy is set to "deepspeed".

    Args:
        config_obj (LLMModelConfig): The configuration object containing backend and optimization information.

    Returns:
        Union[int, None]: The DeepSpeed optimization stage (an integer), or None if not applicable.
    """
    backend_type = _get_backend_type_from_config(config_obj)
    if backend_type != "ray":
        return None

    backend_trainer_config = config_obj.backend.get("trainer", {})
    strategy_config = backend_trainer_config.get("strategy", {})
    if strategy_config.get("type") != "deepspeed":
        return None

    return strategy_config.get("zero_optimization", {}).get("stage", 3)
