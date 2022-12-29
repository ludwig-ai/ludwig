from typing import Type

from ludwig.distributed.base import DistributedStrategy, LocalStrategy
from ludwig.distributed.ddp import DDPStrategy
from ludwig.distributed.horovod import HorovodStrategy


STRATEGIES = {"ddp": DDPStrategy, "horovod": HorovodStrategy, "local": LocalStrategy}


def get_current_dist_strategy() -> Type[DistributedStrategy]:
    for strategy_cls in STRATEGIES.values():
        if strategy_cls.is_available():
            return strategy_cls

    return LocalStrategy


def get_strategy(name: str) -> Type[DistributedStrategy]:
    return STRATEGIES[name]
