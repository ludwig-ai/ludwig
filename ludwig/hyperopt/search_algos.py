import logging
from abc import ABC
from importlib import import_module
from typing import Dict

from ludwig.hyperopt.registry import register_search_algorithm

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


@register_search_algorithm("random")
@register_search_algorithm("variant_generator")
class BasicVariantSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


@register_search_algorithm("hyperopt")
class HyperoptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hyperopt", "hyperopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


@register_search_algorithm("bohb")
class BOHBSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hpbandster", "bohb")
        _is_package_installed("ConfigSpace", "bohb")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"


@register_search_algorithm("ax")
class AxSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("sqlalchemy", "ax")
        _is_package_installed("ax", "ax")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


@register_search_algorithm("bayesopt")
class BayesOptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("bayes_opt", "bayesopt")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state"


@register_search_algorithm("blendsearch")
class BlendsearchSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "blendsearch")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


@register_search_algorithm("cfo")
class CFOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("flaml", "cfo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


@register_search_algorithm("dragonfly")
class DragonflySA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("dragonfly", "dragonfly")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


@register_search_algorithm("hebo")
class HEBOSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("hebo", "hebo")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "random_state_seed"


@register_search_algorithm("skopt")
class SkoptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("skopt", "skopt")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


@register_search_algorithm("nevergrad")
class NevergradSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("nevergrad", "nevergrad")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass


@register_search_algorithm("optuna")
class OptunaSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("optuna", "optuna")
        super().__init__(search_alg_dict)
        self.random_seed_attribute_name = "seed"


@register_search_algorithm("zoopt")
class ZooptSA(SearchAlgorithm):
    def __init__(self, search_alg_dict: Dict) -> None:
        _is_package_installed("zoopt", "zoopt")
        super().__init__(search_alg_dict)

    # override parent method, this search algorithm does not support
    # setting random seed
    def check_for_random_seed(self, ludwig_random_seed: int) -> None:
        pass
