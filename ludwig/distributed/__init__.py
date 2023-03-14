from typing import Type

from ludwig.distributed.base import DistributedStrategy, LocalStrategy


def load_ddp():
    from ludwig.distributed.ddp import DDPStrategy

    return DDPStrategy


def load_fsdp():
    from ludwig.distributed.fsdp import FSDPStrategy

    return FSDPStrategy


def load_horovod():
    from ludwig.distributed.horovod import HorovodStrategy

    return HorovodStrategy


def load_local():
    return LocalStrategy


STRATEGIES = {"ddp": load_ddp, "fsdp": load_fsdp, "horovod": load_horovod, "local": load_local}


def get_current_dist_strategy(allow_local=True) -> Type[DistributedStrategy]:
    for strategy_loader in STRATEGIES.values():
        try:
            strategy_cls = strategy_loader()
        except ImportError:
            continue

        if strategy_cls.is_available():
            return strategy_cls

    if allow_local:
        return LocalStrategy

    raise RuntimeError("Expected current distributed strategy, but none is available")


def get_dist_strategy(name: str) -> Type[DistributedStrategy]:
    return STRATEGIES[name]()


def get_default_strategy_name() -> str:
    try:
        # Use horovod by default for now if it's available
        load_horovod()
        return "horovod"
    except ImportError:
        pass

    # Fallback to DDP if not
    return "ddp"
