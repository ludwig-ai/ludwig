from typing import Dict

from ludwig.constants import TYPE
from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()


def register_search_algorithm(name: str):
    def wrap(cls):
        search_algorithm_registry[name] = cls
        return cls

    return wrap


def get_search_algorithm_cls(name: str):
    return search_algorithm_registry[name]


def instantiate_search_algorithm(search_alg: Dict):
    search_alg_type = search_alg[TYPE]
    cls = get_search_algorithm_cls(search_alg_type)
    return cls(search_alg)
