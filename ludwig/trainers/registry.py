from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import DEFAULT_KEYS, Registry

_trainers_registry = Registry()
_ray_trainers_registry = Registry()


@DeveloperAPI
def get_trainers_registry() -> Registry:
    return _trainers_registry


@DeveloperAPI
def get_ray_trainers_registry() -> Registry:
    return _ray_trainers_registry


@DeveloperAPI
def register_trainer(model_type: str, default=False):
    """Register a trainer class that supports training the given model types.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        model_type: The model_type which dictates the trainer type to use.
        default: Whether the trainer should be the default trainer for the model type.
    """

    def wrap(cls):
        _trainers_registry[model_type] = cls
        if default:
            if DEFAULT_KEYS[0] in _trainers_registry:
                raise ValueError(f"Default trainer already registered for model type {model_type}")
            for key in DEFAULT_KEYS:
                _trainers_registry[key] = cls
        return cls

    return wrap


@DeveloperAPI
def register_ray_trainer(model_type: str, default=False):
    """Register a trainer class that supports training the given model types with Ray backend.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        model_type: The model_type which dictates the trainer type to use.
        default: Whether the trainer should be the default trainer for the model type.
    """

    def wrap(cls):
        _ray_trainers_registry[model_type] = cls
        if default:
            if DEFAULT_KEYS[0] in _ray_trainers_registry:
                raise ValueError(f"Default trainer already registered for model type {model_type}")
            for key in DEFAULT_KEYS:
                _ray_trainers_registry[key] = cls
        return cls

    return wrap
