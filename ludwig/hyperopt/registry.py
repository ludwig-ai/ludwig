from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()


def register_search_algorithm(name: str):
    def wrap(cls):
        search_algorithm_registry[name] = cls
        return cls

    return wrap


def get_search_algorithm_cls(name: str):
    return search_algorithm_registry[name]
