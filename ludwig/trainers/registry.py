from typing import List, Union

from ludwig.utils.registry import DEFAULT_KEYS, Registry

trainers_registry = Registry()
ray_trainers_registry = Registry()


def register_trainer(name: str, model_types: Union[str, List[str]], default=False):
    """Register a trainer class that supports training the given model types.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        name: The name of the trainer, as it can be used in the config.
        model_types: The model types that the trainer supports.
        default: Whether the trainer should be the default trainer for the model type.
    """
    if isinstance(model_types, str):
        model_types = [model_types]

    def wrap(cls):
        for model_type in model_types:
            _model_type_registry = trainers_registry.get(model_type, {})
            _model_type_registry[name] = cls
            if default:
                if DEFAULT_KEYS[0] in _model_type_registry:
                    raise ValueError(f"Default trainer already registered for model type {model_type}")
                for key in DEFAULT_KEYS:
                    _model_type_registry[key] = cls
            trainers_registry[model_type] = _model_type_registry
        return cls

    return wrap


def register_ray_trainer(name: str, model_types: Union[str, List[str]], default=False):
    """Register a trainer class that supports training the given model types with Ray backend.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        name: The name of the trainer, as it can be used in the config.
        model_types: The model types that the trainer supports.
        default: Whether the trainer should be the default trainer for the model type.
    """
    if isinstance(model_types, str):
        model_types = [model_types]

    def wrap(cls):
        for model_type in model_types:
            _model_type_registry = ray_trainers_registry.get(model_type, {})
            _model_type_registry[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    _model_type_registry[key] = cls
            ray_trainers_registry[model_type] = _model_type_registry
        return cls

    return wrap
