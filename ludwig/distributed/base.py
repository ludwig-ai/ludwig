import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from ray.train.backend import BackendConfig
    from ray.train.data_parallel_trainer import DataParallelTrainer


class DistributedStrategy(ABC):
    """Interface that wraps a distributed training framework (Horovod, DDP).

    Distributed strategies modify the model and/or optimizer to coordinate gradient updates among multiple workers
    running in parallel. In most cases, these are using collective communication libraries pass messages between
    processes.
    """

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def wrap_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def rank(self) -> int:
        pass

    @abstractmethod
    def local_size(self) -> int:
        pass

    @abstractmethod
    def local_rank(self) -> int:
        pass

    @abstractmethod
    def barrier(self):
        pass

    @abstractmethod
    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sync_model(self, model: nn.Module):
        pass

    @abstractmethod
    def sync_optimizer(self, optimizer: Optimizer):
        pass

    @abstractmethod
    def broadcast_object(self, v: Any, name: Optional[str] = None) -> Any:
        pass

    @abstractmethod
    def wait_optimizer_synced(self, optimizer: Optimizer):
        pass

    @abstractmethod
    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def gather_all_tensors_fn(cls) -> Optional[Callable]:
        pass

    @classmethod
    @abstractmethod
    def get_ray_trainer_backend(cls, **kwargs) -> Optional[Any]:
        pass

    @classmethod
    @abstractmethod
    def get_trainer_cls(cls, backend_config: "BackendConfig") -> Tuple[Type["DataParallelTrainer"], Dict[str, Any]]:
        pass

    @abstractmethod
    def shutdown(self):
        pass

    def return_first(self, fn: Callable) -> Callable:
        """Wraps function so results are only returned by the first (coordinator) rank.

        The purpose of this function is to reduce network overhead.
        """

        def wrapped(*args, **kwargs):
            res = fn(*args, **kwargs)
            return res if self.rank() == 0 else None

        return wrapped


class LocalStrategy(DistributedStrategy):
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def wrap_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        return optimizer

    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def local_size(self) -> int:
        return 0

    def local_rank(self) -> int:
        return 0

    def barrier(self):
        pass

    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def sync_model(self, model: nn.Module):
        pass

    def sync_optimizer(self, optimizer: Optimizer):
        pass

    def broadcast_object(self, v: Any, name: Optional[str] = None) -> Any:
        return v

    def wait_optimizer_synced(self, optimizer: Optimizer):
        pass

    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        yield

    @classmethod
    def is_available(cls) -> bool:
        # While this strategy is always an option, it is not "distributed" which is the meaning of availability
        # in this context.
        return False

    @classmethod
    def gather_all_tensors_fn(cls) -> Optional[Callable]:
        return None

    @classmethod
    def get_ray_trainer_backend(cls, **kwargs) -> Optional[Any]:
        return None

    @classmethod
    def get_trainer_cls(cls, backend_config: "BackendConfig") -> Tuple[Type["DataParallelTrainer"], Dict[str, Any]]:
        raise ValueError("Cannot construct a trainer from a local strategy.")

    def shutdown(self):
        pass
