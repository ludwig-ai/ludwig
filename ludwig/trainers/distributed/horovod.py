from typing import Any

import horovod.torch as hvd
import torch
from torch import nn
from torch.optim import Optimizer

from ludwig.trainers.distributed.base import DistributedStrategy


class HorovodStrategy(DistributedStrategy):
    def wrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def wrap_optimizer(self, optimizer: Optimizer) -> Optimizer:
        return optimizer

    def size(self) -> int:
        return hvd.size()

    def rank(self) -> int:
        return hvd.rank()

    def barrier(self):
        return hvd.allreduce(torch.as_tensor([0], dtype=torch.int))

    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        return hvd.allreduce(t)

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        return hvd.broadcast(t, root_rank=0)

    def sync_model(self, model: nn.Module):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    def sync_optimizer(self, optimizer: Optimizer):
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def broadcast_object(self, v: Any) -> Any:
        return hvd.broadcast_object(v)
