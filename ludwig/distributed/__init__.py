from typing import Any

from ludwig.distributed.base import DistributedStrategy, LocalStrategy


def load_local():
    return LocalStrategy


def load_accelerate():
    from ludwig.distributed.accelerate import AccelerateStrategy

    return AccelerateStrategy


STRATEGIES = {
    "accelerate": load_accelerate,
    "local": load_local,
    # Legacy aliases for backward compatibility
    "ddp": load_accelerate,
    "fsdp": load_accelerate,
    "deepspeed": load_accelerate,
}


_current_strategy: DistributedStrategy = None


def init_dist_strategy(strategy: str | dict[str, Any], **kwargs) -> DistributedStrategy:
    global _current_strategy
    if isinstance(strategy, dict):
        dtype = strategy.pop("type", None)
        obj = get_dist_strategy(dtype)(**strategy)
    else:
        obj = get_dist_strategy(strategy)(**kwargs)
    _current_strategy = obj
    return obj


def get_current_dist_strategy() -> DistributedStrategy:
    if _current_strategy is None:
        raise RuntimeError("Distributed strategy not initialized")
    return _current_strategy


def get_dist_strategy(strategy: str | dict[str, Any]) -> type[DistributedStrategy]:
    name = strategy
    if isinstance(strategy, dict):
        name = strategy["type"]
    return STRATEGIES[name]()


def get_default_strategy_name() -> str:
    return "accelerate"
