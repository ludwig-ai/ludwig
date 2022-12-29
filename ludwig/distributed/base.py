import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer


class DistributedStrategy(ABC):
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
    def broadcast_object(self, v: Any) -> Any:
        pass

    @abstractmethod
    def wait_optimizer_synced(self, optimizer: Optimizer):
        pass

    @abstractmethod
    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        pass

    @abstractmethod
    @classmethod
    def is_available(cls) -> bool:
        pass

    @abstractmethod
    @classmethod
    def gather_all_tensors_fn(cls) -> Optional[Callable]:
        pass


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

    def broadcast_object(self, v: Any) -> Any:
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
