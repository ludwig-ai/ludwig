import logging
from abc import ABC
from importlib import import_module
from typing import Dict

from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


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


class BasicVariantSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


class HyperoptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hyperopt", "hyperopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


class BOHBSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hpbandster", "bohb")
        _is_package_installed("ConfigSpace", "bohb")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"


class AxSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("sqlalchemy", "ax")
        _is_package_installed("ax", "ax")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class BayesOptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("bayes_opt", "bayesopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


class BlendsearchSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "blendsearch")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class CFOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "cfo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class DragonflySA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("dragonfly", "dragonfly")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


class HEBOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hebo", "hebo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


class SkoptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("skopt", "skopt")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class NevergradSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("nevergrad", "nevergrad")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


class OptunaSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("optuna", "optuna")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"


class ZooptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("zoopt", "zoopt")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


def get_search_algorithm(search_algo):
    return get_from_registry(search_algo, search_algo_registry)


search_algo_registry = {
    None: BasicVariantSA,
    "variant_generator": BasicVariantSA,
    "random": BasicVariantSA,
    "hyperopt": HyperoptSA,
    "bohb": BOHBSA,
    "ax": AxSA,
    "bayesopt": BayesOptSA,
    "blendsearch": BlendsearchSA,
    "cfo": CFOSA,
    "dragonfly": DragonflySA,
    "hebo": HEBOSA,
    "skopt": SkoptSA,
    "nevergrad": NevergradSA,
    "optuna": OptunaSA,
    "zoopt": ZooptSA,
}
