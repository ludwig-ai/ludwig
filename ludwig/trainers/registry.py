from typing import List, Union

from ludwig.utils.registry import DEFAULT_KEYS, Registry

trainers_registry = Registry()
ray_trainers_registry = Registry()


def register_trainer(name: str, model_types: Union[str, List[str]], default=False):
    if isinstance(model_types, str):
        model_types = [model_types]

    def wrap(cls):
        for model_type in model_types:
            _model_type_registry = trainers_registry.get(model_type, {})
            _model_type_registry[name] = cls
            if default:
                for key in DEFAULT_KEYS:
                    _model_type_registry[key] = cls
            trainers_registry[model_type] = _model_type_registry
        return cls

    return wrap


def register_ray_trainer(name: str, model_types: Union[str, List[str]], default=False):
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
