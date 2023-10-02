import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

import torch
from torch import nn
from torch.optim import Optimizer

from ludwig.modules.optimization_modules import create_optimizer
from ludwig.utils.torch_utils import get_torch_device

if TYPE_CHECKING:
    from ray.train.backend import BackendConfig
    from ray.train.data_parallel_trainer import DataParallelTrainer

    from ludwig.models.base import BaseModel
    from ludwig.modules.lr_scheduler import LRScheduler
    from ludwig.schema.trainer import ECDTrainerConfig
    from ludwig.utils.checkpoint_utils import Checkpoint


class DistributedStrategy(ABC):
    """Interface that wraps a distributed training framework (Horovod, DDP).

    Distributed strategies modify the model and/or optimizer to coordinate gradient updates among multiple workers
    running in parallel. In most cases, these are using collective communication libraries pass messages between
    processes.
    """

    @abstractmethod
    def prepare(
        self,
        model: nn.Module,
        trainer_config: "ECDTrainerConfig",
        base_learning_rate: float,
    ) -> Tuple[nn.Module, Optimizer]:
        """Modifies the model to support distributed training and creates the optimizer.

        Args:
            model: The model to wrap for distributed training.
            trainer_config: The trainer configuration, which includes optimizer params.
            base_learning_rate: The base learning rate to init the optimizer, which may be scaled by the strategy.

        Returns:
            A tuple of the wrapped model and the optimizer.
        """
        pass

    def prepare_for_inference(self, model: nn.Module) -> nn.Module:
        return model

    def to_device(self, model: "BaseModel", device: Optional[torch.device] = None) -> nn.Module:
        return model.to_device(device if device is not None else get_torch_device())

    def backward(self, loss: torch.Tensor, model: nn.Module):
        loss.backward()

    def step(self, optimizer: Optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def zero_grad(self, optimizer: Optimizer):
        optimizer.zero_grad()

    def set_batch_size(self, model: nn.Module, batch_size: int):
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

    def is_coordinator(self) -> bool:
        return self.rank() == 0

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
    def prepare_model_update(self, model: nn.Module, should_step: bool):
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

    def allow_gradient_accumulation(self) -> bool:
        return True

    def allow_mixed_precision(self) -> bool:
        return True

    def allow_clip_gradients(self) -> bool:
        return True

    def prepare_before_load(self) -> bool:
        """True if we need to call `prepare` again before loading a checkpoint."""
        return False

    @classmethod
    def is_model_parallel(cls) -> bool:
        return False

    def create_checkpoint_handle(
        self,
        dist_model: nn.Module,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional["LRScheduler"] = None,
    ) -> "Checkpoint":
        from ludwig.utils.checkpoint_utils import MultiNodeCheckpoint

        return MultiNodeCheckpoint(self, model, optimizer, scheduler)

    @classmethod
    def extract_model_for_serialization(cls, model: nn.Module) -> Union[nn.Module, Tuple[nn.Module, List[Dict]]]:
        return model

    @classmethod
    def replace_model_from_serialization(cls, state: Union[nn.Module, Tuple[nn.Module, List[Dict]]]) -> nn.Module:
        assert isinstance(state, nn.Module)
        return state


class LocalStrategy(DistributedStrategy):
    def prepare(
        self,
        model: nn.Module,
        trainer_config: "ECDTrainerConfig",
        base_learning_rate: float,
    ) -> Tuple[nn.Module, Optimizer]:
        return model, create_optimizer(model, trainer_config.optimizer, base_learning_rate)

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
    def prepare_model_update(self, model: nn.Module, should_step: bool):
        yield

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
