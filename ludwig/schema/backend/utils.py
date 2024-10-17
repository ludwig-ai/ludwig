from typing import Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry

backend_config_registry = Registry()
loader_config_registry = Registry()
processor_config_registry = Registry()
trainer_config_registry = Registry()


@DeveloperAPI
def get_backend_config(name: str) -> Type["BaseBackendConfig"]:  # noqa: F821
    """Get a registered backend config class by name.

    Args:
        name: the name of a backend config class registered in `ludwig.schema.backend`

    Returns:
        A backend config class from `ludwig.schema.backend`
    """
    return backend_config_registry[name]


@DeveloperAPI
def get_processor_config(name: str) -> Type["BaseProcessorConfig"]:  # noqa: F821
    """Get a registered backend config class by name.

    Args:
        name: the name of a backend config class registered in `ludwig.schema.backend.processor`

    Returns:
        A backend config class from `ludwig.schema.backend.processor`
    """
    return processor_config_registry[name]


@DeveloperAPI
def register_backend_config(name: str):
    """Register a backend config class by name.

    Args:
        name: the name to register the backend class under, does not need to correspond to the value of `type`

    Returns:
        Wrapper function to decorate a `BaseBackendConfig` subclass
    """

    def wrap(cls: Type["BaseBackendConfig"]) -> Type["BaseBackendConfig"]:  # noqa: F821
        """Add a backend config class to the registry.

        Args:
            cls: a subclass of `BaseBackendConfig`

        Returns:
            `cls` unaltered
        """
        backend_config_registry[name] = cls
        return cls

    return wrap


@DeveloperAPI
def register_processor_config(name: str):
    """Register a scheduler config class by name.

    Args:
        name: the name to scheduler the backend class under, does not need to correspond to the value of `type`

    Returns:
        Wrapper function to decorate a `BaseBackendConfig` subclass
    """

    def wrap(cls: Type["BaseProcessorConfig"]) -> Type["BaseProcessorConfig"]:  # noqa: F821
        """Add a backend config class to the registry.

        Args:
            cls: a subclass of `BaseProcessorConfig`

        Returns:
            `cls` unaltered
        """
        processor_config_registry[name] = cls
        return cls

    return wrap
