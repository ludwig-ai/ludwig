from typing import Callable, List, Optional, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry

parameter_type_registry = Registry()
scheduler_config_registry = Registry()
scheduler_dependencies_registry = Registry()
search_algorithm_registry = Registry()
sa_dependencies_registry = Registry()


@DeveloperAPI
def get_parameter_cls(name: str) -> Type["BaseParameterConfig"]:  # noqa: F821
    """Get a registered hyperopt parameter config class by name.

    Args:
        name: the name of a parameter config class registered in `ludwig.schema.hyperopt.parameter`

    Returns:
        A parameter config class from `ludwig.schema.hyperopt.parameter`
    """
    return parameter_type_registry.get[name]


@DeveloperAPI
def get_scheduler_cls(name: str) -> Type["BaseSchedulerConfig"]:  # noqa: F821
    """Get a registered hyperopt scheduler config class by name.

    Args:
        name: the name of a scheduler config class registered in `ludwig.schema.hyperopt.scheduler`

    Returns:
        A scheduler config class from `ludwig.schema.hyperopt.scheduler`
    """
    return search_algorithm_registry[name]


@DeveloperAPI
def get_scheduler_dependencies(name: str) -> List[str]:
    """Get the list of dependencies for a registered hyperopt scheduler.

    Args:
        name: the name of a scheduler config class registered in `ludwig.schema.hyperopt.scheduler`

    Returns:
        The list of imports needed to use the scheduler
    """
    return scheduler_dependencies_registry[name]


@DeveloperAPI
def get_search_algorithm_cls(name: str) -> Type["BaseSearchAlgorithmConfig"]:  # noqa: F821
    """Get a registered hyperopt search algorithm config class by name.

    Args:
        name: the name of a search algorithm config class registered in `ludwig.schema.hyperopt.search_algorithm`

    Returns:
        A scheduler config class from `ludwig.schema.hyperopt.search_algorithm`
    """
    return search_algorithm_registry[name]


@DeveloperAPI
def get_search_algorithm_dependencies(name: str) -> List[str]:
    """Get the list of dependencies for a registered hyperopt search algorithm.

    Args:
        name: the name of a search algorithm config class registered in `ludwig.schema.hyperopt.search_algorithm`

    Returns:
        The list of imports needed to use the search algorithm
    """
    return sa_dependencies_registry[name]


@DeveloperAPI
def register_parameter_config(name: str) -> Callable:
    """Register a parameter config class by name.

    Args:
        name: the name to register the parameter class under, does not need to correspond to the value of `space`

    Returns:
        Wrapper function to decorate a `BaseParameterConfig` subclass
    """

    def wrap(cls: Type["BaseParameterConfig"]) -> Type["BaseParameterConfig"]:  # noqa: F821
        """Add a parameter config class to the registry.

        Args:
            cls: a subclass of `BaseParameterConfig`

        Returns:
            `cls` unaltered
        """
        parameter_type_registry[name] = cls
        return cls

    return wrap


@DeveloperAPI
def register_scheduler_config(name: str, dependencies: Optional[List[str]] = None):
    """Register a scheduler config class by name.

    Args:
        name: the name to scheduler the parameter class under, does not need to correspond to the value of `type`
        dependencies: the list of module names that the scheduler requires

    Returns:
        Wrapper function to decorate a `BaseSchedulerConfig` subclass
    """

    def wrap(scheduler_config: Type["BaseSchedulerConfig"]) -> Type["BaseSchedulerConfig"]:  # noqa: F821
        """Add a parameter config class to the registry.

        Args:
            cls: a subclass of `BaseParameterConfig`

        Returns:
            `cls` unaltered
        """
        scheduler_config_registry[name] = scheduler_config
        scheduler_dependencies_registry[name] = dependencies if dependencies is not None else []
        return scheduler_config

    return wrap


@DeveloperAPI
def register_search_algorithm(name: str, dependencies: Optional[List[str]] = None) -> Callable:
    """Register a search algorithm config class by name.

    Args:
        name: the name to register the search algorithm class under, does not need to correspond to the value of `type`
        dependencies: the list of module names that the search algorithm requires

    Returns:
        Wrapper function to decorate a
    """

    def wrap(cls: Type["BaseSearchAlgorithmConfig"]) -> Type["BaseSearchAlgorithmConfig"]:  # noqa: F821
        search_algorithm_registry[name] = cls
        sa_dependencies_registry[name] = dependencies if dependencies is not None else []
        return cls

    return wrap
