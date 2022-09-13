from ludwig.utils.registry import Registry

search_algorithm_registry = Registry()


def register_search_algorithm(name: str):
    print("test")

    def wrap(cls):
        search_algorithm_registry[name] = cls
        return cls

    return wrap
