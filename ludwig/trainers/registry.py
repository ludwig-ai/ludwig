from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import DEFAULT_KEYS, Registry

_trainers_registry = Registry()
_ray_trainers_registry = Registry()

_llm_trainers_registry = Registry()
_llm_ray_trainers_registry = Registry()


@DeveloperAPI
def get_trainers_registry() -> Registry:
    return _trainers_registry


@DeveloperAPI
def get_ray_trainers_registry() -> Registry:
    return _ray_trainers_registry


@DeveloperAPI
def get_llm_trainers_registry() -> Registry:
    return _llm_trainers_registry


@DeveloperAPI
def get_llm_ray_trainers_registry() -> Registry:
    return _llm_ray_trainers_registry


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


@DeveloperAPI
def register_llm_trainer(trainer_type: str, default=False):
    """Register a trainer class that supports training the specific type of training strategy for LLM Models.

    Using default=True will make the trainer the default trainer for the LLM model type.

    Args:
        trainer_type: The trainer_type which dictates what training strategy to use.
        default: Whether the trainer should be the default trainer for LLMs.
    """

    def wrap(cls):
        _llm_trainers_registry[trainer_type] = cls
        if default:
            if DEFAULT_KEYS[0] in _trainers_registry:
                raise ValueError(f"Default trainer {trainer_type} already registered for LLM")
            for key in DEFAULT_KEYS:
                _llm_trainers_registry[key] = cls
        return cls

    return wrap


@DeveloperAPI
def register_llm_ray_trainer(trainer_type: str, default=False):
    """Register a trainer class that supports training the specific type of training strategy for LLM Models with
    Ray backend.

    Using default=True will make the trainer the default trainer for the LLM model type.

    Args:
        trainer_type: The trainer_type which dictates what training strategy to use.
        default: Whether the trainer should be the default trainer for LLMs.
    """

    def wrap(cls):
        _llm_ray_trainers_registry[trainer_type] = cls
        if default:
            if DEFAULT_KEYS[0] in _trainers_registry:
                raise ValueError(f"Default ray trainer {trainer_type} already registered for LLM")
            for key in DEFAULT_KEYS:
                _llm_ray_trainers_registry[key] = cls
        return cls

    return wrap
