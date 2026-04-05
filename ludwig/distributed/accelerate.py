"""HuggingFace Accelerate distributed training strategy.

Provides a single abstraction for DDP, FSDP, and DeepSpeed via the Accelerate library.
This replaces the need for separate DDPStrategy, FSDPStrategy, and DeepSpeedStrategy
classes with one unified implementation.

Usage:
    # In config or backend:
    strategy: accelerate

    # With FSDP:
    strategy:
      type: accelerate
      mixed_precision: bf16
      fsdp_config:
        fsdp_sharding_strategy: FULL_SHARD

    # With DeepSpeed:
    strategy:
      type: accelerate
      deepspeed_config: path/to/ds_config.json
"""

import contextlib
import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.optim import Optimizer

from ludwig.distributed.base import DistributedStrategy
from ludwig.modules.optimization_modules import create_optimizer
from ludwig.utils.checkpoint_utils import Checkpoint, MultiNodeCheckpoint

logger = logging.getLogger(__name__)


class AccelerateStrategy(DistributedStrategy):
    """Distributed training via HuggingFace Accelerate.

    Accelerate provides a single prepare() call that auto-detects and configures DDP, FSDP, or DeepSpeed based on the
    environment and configuration.
    """

    def __init__(
        self,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int = 1,
        fsdp_config: dict | None = None,
        deepspeed_config: str | dict | None = None,
        **kwargs,
    ):
        from accelerate import Accelerator

        accelerator_kwargs = {}
        if mixed_precision:
            accelerator_kwargs["mixed_precision"] = mixed_precision
        if gradient_accumulation_steps > 1:
            accelerator_kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps
        if fsdp_config:
            # Accelerate FSDP config is passed via fsdp_plugin
            from accelerate import FullyShardedDataParallelPlugin

            accelerator_kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(**fsdp_config)
        if deepspeed_config:
            from accelerate import DeepSpeedPlugin

            if isinstance(deepspeed_config, str):
                accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=deepspeed_config)
            else:
                accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(**deepspeed_config)

        self.accelerator = Accelerator(**accelerator_kwargs)
        self._prepared_model = None
        self._prepared_optimizer = None

        logger.info(
            f"AccelerateStrategy initialized: distributed_type={self.accelerator.distributed_type}, "
            f"num_processes={self.accelerator.num_processes}, device={self.accelerator.device}"
        )

    def prepare(
        self,
        model: Module,
        trainer_config,
        base_learning_rate: float,
    ) -> tuple[Module, Optimizer]:
        optimizer = create_optimizer(model, trainer_config.optimizer, base_learning_rate)
        self._prepared_model, self._prepared_optimizer = self.accelerator.prepare(model, optimizer)
        return self._prepared_model, self._prepared_optimizer

    def to_device(self, model, device=None):
        # Accelerate handles device placement via prepare()
        if device is not None:
            return model.to_device(device)
        return model.to_device(self.accelerator.device)

    def backward(self, loss: torch.Tensor, model: Module):
        self.accelerator.backward(loss)

    def step(self, optimizer: Optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def zero_grad(self, optimizer: Optimizer):
        optimizer.zero_grad()

    def size(self) -> int:
        return self.accelerator.num_processes

    def rank(self) -> int:
        return self.accelerator.process_index

    def local_size(self) -> int:
        # Accelerate doesn't expose local_size directly. Compute from environment.
        return int(os.environ.get("LOCAL_SIZE", os.environ.get("LOCAL_WORLD_SIZE", 1)))

    def local_rank(self) -> int:
        return self.accelerator.local_process_index

    def barrier(self):
        self.accelerator.wait_for_everyone()

    def allreduce(self, t: torch.Tensor) -> torch.Tensor:
        return self.accelerator.reduce(t, reduction="sum")

    def broadcast(self, t: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(t, src=0)
        return t

    def sync_model(self, model: Module):
        # Accelerate handles model sync through prepare()
        pass

    def sync_optimizer(self, optimizer: Optimizer):
        # Accelerate handles optimizer sync through prepare()
        pass

    def broadcast_object(self, v: Any, name: str | None = None) -> Any:
        if not dist.is_available() or not dist.is_initialized():
            return v
        obj_list = [v]
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]

    def wait_optimizer_synced(self, optimizer: Optimizer):
        # Accelerate handles gradient sync automatically
        pass

    @contextlib.contextmanager
    def prepare_model_update(self, model: Module, should_step: bool):
        # Emulate DDP's no_sync() behavior for gradient accumulation.
        # When should_step is False, we skip gradient synchronization.
        if not should_step and hasattr(model, "no_sync"):
            with model.no_sync():
                yield
        else:
            yield

    @contextlib.contextmanager
    def prepare_optimizer_update(self, optimizer: Optimizer):
        yield

    @classmethod
    def is_available(cls) -> bool:
        try:
            import accelerate  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def gather_all_tensors_fn(cls):
        try:
            from torchmetrics.utilities.distributed import gather_all_tensors

            return gather_all_tensors
        except ImportError:
            return None

    @classmethod
    def get_ray_trainer_backend(cls, **kwargs):
        # Accelerate can work with Ray but requires specific setup.
        # Return TorchConfig for basic compatibility.
        try:
            from ray.train.torch import TorchConfig

            return TorchConfig()
        except ImportError:
            return None

    @classmethod
    def get_trainer_cls(cls, backend_config):
        try:
            from ray.train.torch import TorchTrainer

            return TorchTrainer, {"torch_config": backend_config}
        except ImportError:
            raise ImportError("Ray is required for distributed training with AccelerateStrategy via Ray backend.")

    def shutdown(self):
        pass

    def prepare_for_inference(self, model: Module) -> Module:
        return self.accelerator.unwrap_model(model)

    def allow_gradient_accumulation(self) -> bool:
        return True

    def allow_mixed_precision(self) -> bool:
        # Let Accelerate handle mixed precision if configured
        if self.accelerator.mixed_precision != "no":
            return False
        return True

    def allow_clip_gradients(self) -> bool:
        return True

    def prepare_before_load(self) -> bool:
        return False

    @classmethod
    def is_model_parallel(cls) -> bool:
        return False

    def create_checkpoint_handle(self, dist_model, model, optimizer, scheduler) -> Checkpoint:
        return MultiNodeCheckpoint(self, model, optimizer, scheduler)

    @classmethod
    def extract_model_for_serialization(cls, model):
        try:
            from accelerate import Accelerator

            accelerator = Accelerator()
            return accelerator.unwrap_model(model)
        except Exception:
            return model

    @classmethod
    def replace_model_from_serialization(cls, state):
        assert isinstance(state, Module)
        return state
