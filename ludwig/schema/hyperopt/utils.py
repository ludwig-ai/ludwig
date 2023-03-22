from typing import Callable, List, Optional, Tuple, Type

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils.registry import Registry

parameter_config_registry = Registry()
scheduler_config_registry = Registry()
scheduler_dependencies_registry = Registry()
search_algorithm_config_registry = Registry()
search_algorithm_dependencies_registry = Registry()
search_algorithm_random_state_field_registry = Registry()


@DeveloperAPI
def get_parameter_cls(name: str) -> Type["BaseParameterConfig"]:  # noqa: F821
    """Get a registered hyperopt parameter config class by name.

    Args:
        name: the name of a parameter config class registered in `ludwig.schema.hyperopt.parameter`

    Returns:
        A parameter config class from `ludwig.schema.hyperopt.parameter`
    """
    return parameter_config_registry[name]


@DeveloperAPI
def get_scheduler_cls(name: str) -> Type["BaseSchedulerConfig"]:  # noqa: F821
    """Get a registered hyperopt scheduler config class by name.

    Args:
        name: the name of a scheduler config class registered in `ludwig.schema.hyperopt.scheduler`

    Returns:
        A scheduler config class from `ludwig.schema.hyperopt.scheduler`
    """
    return search_algorithm_config_registry[name]


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
    return search_algorithm_config_registry[name]


@DeveloperAPI
def get_search_algorithm_dependencies(name: str) -> List[str]:
    """Get the list of dependencies for a registered hyperopt search algorithm.

    Args:
        name: the name of a search algorithm config class registered in `ludwig.schema.hyperopt.search_algorithm`

    Returns:
        The list of imports needed to use the search algorithm
    """
    return search_algorithm_dependencies_registry[name]


@DeveloperAPI
def get_search_algorithm_random_state_field(name: str):
    """Get the field name of the random state for a registered hyperopt search algorithm.

    Args:
        name: the name of a search algorithm config class registered in `ludwig.schema.hyperopt.search_algorithm`

    Returns:
        The name of the random state field in the config
    """
    return search_algorithm_random_state_field_registry[name]


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
        parameter_config_registry[name] = cls
        return cls

    return wrap


@DeveloperAPI
def register_scheduler_config(name: str, dependencies: Optional[List[Tuple[str]]] = None):
    """Register a scheduler config class by name.

    Args:
        name: the name to scheduler the parameter class under, does not need to correspond to the value of `type`
        dependencies: the list of scheduler dependency package name/install name pairs, e.g.
                      `("sklearn", "scikit-learn")`

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


# TODO: create a search alg metadata class to register in place of individual metadata args
@DeveloperAPI
def register_search_algorithm_config(
    name: str, random_state_field: Optional[str] = None, dependencies: Optional[List[Tuple[str, str]]] = None
) -> Callable:
    """Register a search algorithm config class by name.

    Args:
        name: the name to register the search algorithm class under, does not need to correspond to the value of `type`
        random_state_field: the name of the random state in this search algorithm
        dependencies: the list of search algorithm dependency package name/install name pairs, e.g.
                      `("sklearn", "scikit-learn")`

    Returns:
        Wrapper function to decorate a `BaseSearchAlgorithmConfig` subclass
    """

    def wrap(cls: Type["BaseSearchAlgorithmConfig"]) -> Type["BaseSearchAlgorithmConfig"]:  # noqa: F821
        search_algorithm_config_registry[name] = cls
        search_algorithm_dependencies_registry[name] = dependencies if dependencies is not None else []
        search_algorithm_random_state_field_registry[name] = random_state_field
        return cls

    return wrap
