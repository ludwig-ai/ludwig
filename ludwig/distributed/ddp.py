import contextlib
import logging
import os
import socket
from typing import Any, Callable, Dict, Optional, Tuple, Type, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
from ray.train.backend import BackendConfig
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.torch import TorchTrainer
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchmetrics.utilities.distributed import gather_all_tensors

from ludwig.distributed.base import DistributedStrategy
from ludwig.modules.optimization_modules import create_optimizer
from ludwig.utils.torch_utils import get_torch_device

if TYPE_CHECKING:
    from ludwig.models.base import BaseModel
    from ludwig.modules.lr_scheduler import LRScheduler
    from ludwig.schema.trainer import ECDTrainerConfig
    from ludwig.utils.checkpoint_utils import Checkpoint


class DDPStrategy(DistributedStrategy):
    def __init__(self):
        self._local_rank, self._local_size = local_rank_and_size()
        self._log_on_init()

    def _log_on_init(self):
        logging.info("Using DDP strategy")

    def prepare(
        self,
        model: nn.Module,
        trainer_config: "ECDTrainerConfig",
        base_learning_rate: float,
    ) -> Tuple[nn.Module, Optimizer]:
        return DDP(model), create_optimizer(model, trainer_config.optimizer, base_learning_rate)

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
        dist.all_reduce(t)
        return t

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        dist.broadcast(t)
        return t

    def sync_model(self, model: nn.Module):
        # TODO(travis): open question if this is needed to ensure all workers using same weights
        pass

    def sync_optimizer(self, optimizer: Optimizer):
        # TODO(travis): open question if this is needed to ensure all workers using same optimizer state
        pass

    def broadcast_object(self, v: Any, name: Optional[str] = None) -> Any:
        output = [v]
        dist.broadcast_object_list(output)
        return output[0]

    def wait_optimizer_synced(self, optimizer: Optimizer):
        pass

    @contextlib.contextmanager
    def prepare_model_update(self, model: nn.Module, should_step: bool):
        if should_step:
            yield
        else:
            # Prevents DDP from syncing gradients during accumulation step
            with model.no_sync():
                yield

    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        yield

    @classmethod
    def is_available(cls) -> bool:
        return dist.is_available() and dist.is_initialized()

    @classmethod
    def gather_all_tensors_fn(cls) -> Optional[Callable]:
        return gather_all_tensors

    @classmethod
    def get_ray_trainer_backend(cls, **kwargs) -> Optional[Any]:
        from ray.train.torch import TorchConfig

        return TorchConfig()

    @classmethod
    def get_trainer_cls(cls, backend_config: BackendConfig) -> Tuple[Type[DataParallelTrainer], Dict[str, Any]]:
        return TorchTrainer, dict(torch_config=backend_config)

    def shutdown(self):
        # TODO(travis): currently Ray handles this for us, but is subject to hangs if one of the workers raises an
        # exception and the other makes a collective op. We should figure out a way to make this safe to call
        # multiple times. It looks like there is a fix we can make use of when we upgrade to Ray 2.1:
        # https://discuss.ray.io/t/torchtrainer-hangs-when-only-1-worker-raises-error/7447/11
        # dist.destroy_process_group()
        pass

    def create_checkpoint_handle(
        self,
        dist_model: nn.Module,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional["LRScheduler"] = None,
    ) -> "Checkpoint":
        from ludwig.utils.checkpoint_utils import MultiNodeCheckpoint

        return MultiNodeCheckpoint(self, model, optimizer, scheduler)

    def to_device(self, model: Union["BaseModel", DDP], device: Optional[torch.device] = None) -> nn.Module:
        try:
            return model.to_device(device if device is not None else get_torch_device())
        except AttributeError:
            # Model is already wrapped in DistributedDataParallel, so it has already been moved to device
            return model


def local_rank_and_size() -> Tuple[int, int]:
    # DeepSpeed CLI and other tools may set these environment variables for us.
    local_rank, local_size = os.environ.get("LOCAL_RANK"), os.environ.get("LOCAL_SIZE")
    if local_rank is not None and local_size is not None:
        return int(local_rank), int(local_size)

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
