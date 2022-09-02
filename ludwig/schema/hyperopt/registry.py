from typing import Type

from ludwig.hyperopt.base import SearchAlgorithm
from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()


def register_search_algorithm(name: str):
    print("test")

    def wrap(cls):
        search_algorithm_registry[name] = cls
        return cls

    return wrap


def get_search_algorithm_cls(name: str) -> Type[SearchAlgorithm]:
    return search_algorithm_registry[name]
