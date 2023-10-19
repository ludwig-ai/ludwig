import os
from typing import Any, Dict, TYPE_CHECKING, Union

if TYPE_CHECKING:
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


def _get_backend_type_from_config(config_obj: "ModelConfig") -> str:  # noqa: F821
    """Get the backend type from a model configuration object.

    This function retrieves the backend type specified in a language model configuration.

    Args:
        config_obj (ModelConfig): The configuration object containing backend information.

    Returns:
        str: The backend type, defaulting to "local" if not explicitly specified.
    """
    # config_obj.backend may be None if the backend was not explicitly set in the config
    backend = config_obj.backend or {}
    backend_type = backend.get("type", "local")
    return backend_type


def _get_optimization_stage_from_trainer_config(trainer_config: Dict[str, Any]) -> Union[int, None]:
    """Retrieve the DeepSpeed optimization stage from the backend's trainer configuration.

    This function extracts the DeepSpeed optimization stage from the backend's trainer configuration if the
    distributed strategy type is set to "deepspeed".

    Args:
        trainer_config (Dict[str, Any]): The configuration dictionary for the backend's trainer.

    Returns:
        Union[int, None]: The DeepSpeed optimization stage (an integer), or None if DeepSpeed is not the
        selected strategy or no stage is specified.
    """
    # Distributed strategy can directly be the strategy name or be a dict with strategy name and kwargs
    distributed_strategy_name_or_kwargs = trainer_config.get("strategy", "ddp")
    if isinstance(distributed_strategy_name_or_kwargs, dict):
        distributed_strategy_type = distributed_strategy_name_or_kwargs.get("type", "ddp")
    else:
        distributed_strategy_type = distributed_strategy_name_or_kwargs

    if distributed_strategy_type != "deepspeed":
        return None

    if isinstance(distributed_strategy_name_or_kwargs, str):
        return None
    return distributed_strategy_name_or_kwargs.get("zero_optimization", {}).get("stage", 3)


def _get_deepspeed_optimization_stage_from_config(config_obj: "ModelConfig") -> Union[int, None]:  # noqa: F821
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
    return _get_optimization_stage_from_trainer_config(backend_trainer_config)
