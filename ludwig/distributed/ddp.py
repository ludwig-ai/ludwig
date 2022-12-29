import contextlib
import socket
from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from ludwig.distributed.base import DistributedStrategy


class DDPStrategy(DistributedStrategy):
    def __init__(self):
        self._local_rank, self._local_size = local_rank_and_size()

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return DDP(model, device_ids=[self.rank()])

    def wrap_optimizer(self, optimizer: Optimizer, model: nn.Module) -> Optimizer:
        return optimizer

    def size(self) -> int:
        return dist.get_world_size()

    def rank(self) -> int:
        return dist.get_rank()

    def local_size(self) -> int:
        return self._local_size

    def local_rank(self) -> int:
        return self._local_rank

    def barrier(self):
        return dist.barrier()

    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        return dist.all_reduce(t)

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        return dist.broadcast(t)

    def sync_model(self, model: nn.Module):
        # TODO(travis): open question if this is needed to ensure all workers using same weights
        pass

    def sync_optimizer(self, optimizer: Optimizer):
        # TODO(travis): open question if this is needed to ensure all workers using same optimizer state
        pass

    def broadcast_object(self, v: Any) -> Any:
        return dist.broadcast_object_list([v])[0]

    def wait_optimizer_synced(self, optimizer: Optimizer):
        pass

    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        yield


def local_rank_and_size() -> Tuple[int, int]:
    # Gather the rank and hostnames from every worker so we can count up how many belong to the same host, which
    # constitutes the local group.
    rank = dist.get_rank()
    host = socket.gethostname()
    output = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output, (rank, host))

    # Every time we find a worker with the same host, we increment the size counter.
    # The local rank is determined by the world rank relative to the other workers on the same host, so every time
    # we see a worker on our host with a lower rank, we increment the rank counter.
    local_size = 0
    local_rank = 0
    for other_rank, other_host in output:
        if other_host == host:
            local_size += 1
            if other_rank < rank:
                local_rank += 1

    return local_rank, local_size
