from abc import ABC
from importlib import import_module
from typing import Dict


def _is_package_installed(package_name: str, search_algo_name: str) -> bool:
    try:
        import_module(package_name)
        return True
    except ImportError:
        raise ImportError(
            f"Search algorithm {search_algo_name} requires package {package_name}, however package is not installed."
            " Please refer to Ray Tune documentation for packages required for this search algorithm."
        )


class SearchAlgorithm(ABC):
    def __init__(self, search_alg_dict: Dict) -> None:
        self.search_alg_dict = search_alg_dict
        self.random_seed_attribute_name = None

    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        if self.random_seed_attribute_name not in self.search_alg_dict:
            self.search_alg_dict[self.random_seed_attribute_name] = ludwig_random_seed
