from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


class DistributedStrategy(ABC):
    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def wrap_optimizer(self, optimizer: Optimizer) -> Optimizer:
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def rank(self) -> int:
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
