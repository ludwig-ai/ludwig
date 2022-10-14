from ludwig.utils.registry import DEFAULT_KEYS, Registry

trainers_registry = Registry()
ray_trainers_registry = Registry()


def register_trainer(model_type: str, default=False):
    """Register a trainer class that supports training the given model types.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        model_type: The model_type which dictates the trainer type to use.
        default: Whether the trainer should be the default trainer for the model type.
    """

    def wrap(cls):
        trainers_registry[model_type] = cls
        if default:
            if DEFAULT_KEYS[0] in trainers_registry:
                raise ValueError(f"Default trainer already registered for model type {model_type}")
            for key in DEFAULT_KEYS:
                trainers_registry[key] = cls
        return cls

    return wrap


def register_ray_trainer(model_type: str, default=False):
    """Register a trainer class that supports training the given model types with Ray backend.

    Using default=True will make the trainer the default trainer for the model type.

    Args:
        model_type: The model_type which dictates the trainer type to use.
        default: Whether the trainer should be the default trainer for the model type.
    """

    def wrap(cls):
        ray_trainers_registry[model_type] = cls
        if default:
            if DEFAULT_KEYS[0] in ray_trainers_registry:
                raise ValueError(f"Default trainer already registered for model type {model_type}")
            for key in DEFAULT_KEYS:
                ray_trainers_registry[key] = cls
        return cls

    return wrap
