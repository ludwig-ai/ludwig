import logging
from typing import Type

from ludwig.distributed.base import DistributedStrategy, LocalStrategy


def load_ddp():
    from ludwig.distributed.ddp import DDPStrategy

    return DDPStrategy


def load_horovod():
    from ludwig.distributed.horovod import HorovodStrategy

    return HorovodStrategy


def load_local():
    return LocalStrategy


STRATEGIES = {"ddp": load_ddp, "horovod": load_horovod, "local": load_local}


def get_current_dist_strategy(allow_local=True) -> Type[DistributedStrategy]:
    for strategy_name, strategy_loader in STRATEGIES.items():
        try:
            strategy_cls = strategy_loader()
        except ImportError:
            logging.info(f"Distributed strategy {strategy_name} is not available due to import error")
            continue

        if strategy_cls.is_available():
            return strategy_cls

    if allow_local:
        return LocalStrategy

    raise RuntimeError("Expected current distributed strategy, but none is available")


def get_dist_strategy(name: str) -> Type[DistributedStrategy]:
    return STRATEGIES[name]()
