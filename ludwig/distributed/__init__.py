from typing import Any, Dict, Type, Union

from ludwig.distributed.base import DistributedStrategy, LocalStrategy


def load_ddp():
    from ludwig.distributed.ddp import DDPStrategy

    return DDPStrategy


def load_fsdp():
    from ludwig.distributed.fsdp import FSDPStrategy

    return FSDPStrategy


def load_deepspeed():
    from ludwig.distributed.deepspeed import DeepSpeedStrategy

    return DeepSpeedStrategy


def load_horovod():
    from ludwig.distributed.horovod import HorovodStrategy

    return HorovodStrategy


def load_local():
    return LocalStrategy


STRATEGIES = {
    "ddp": load_ddp,
    "fsdp": load_fsdp,
    "deepspeed": load_deepspeed,
    "horovod": load_horovod,
    "local": load_local,
}


_current_strategy: DistributedStrategy = None


def init_dist_strategy(strategy: Union[str, Dict[str, Any]], **kwargs) -> DistributedStrategy:
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


def get_dist_strategy(strategy: Union[str, Dict[str, Any]]) -> Type[DistributedStrategy]:
    name = strategy
    if isinstance(strategy, dict):
        name = strategy["type"]
    return STRATEGIES[name]()


def get_default_strategy_name() -> str:
    return "ddp"
