from ludwig.schema.hyperopt.search_algorithm import get_search_algorithm_cls


def instantiate_search_algorithm(search_alg: "BaseSearchAlgorithmConfig") -> "SearchAlgorithm":  # noqa: F821
    search_alg_type = search_alg.type
    cls = get_search_algorithm_cls(search_alg_type)
    return cls(search_alg)
