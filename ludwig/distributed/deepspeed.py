import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple, TYPE_CHECKING

import deepspeed
import deepspeed.comm
import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch import nn
from torch.optim.optimizer import Optimizer

from ludwig.distributed.ddp import DDPStrategy
from ludwig.utils.checkpoint_utils import Checkpoint

if TYPE_CHECKING:
    from ludwig.modules.lr_scheduler import LRScheduler
    from ludwig.schema.trainer import ECDTrainerConfig


DEFAULT_ZERO_OPTIMIZATION = {
    "stage": "auto",
    "stage3_gather_16bit_weights_on_model_save": "auto",
    "offload_optimizer": {"device": "auto"},
    "offload_param": {"device": "auto"},
}


class DeepSpeedStrategy(DDPStrategy):
    def __init__(self, zero_optimization: Optional[Dict[str, Any]] = None, **kwargs):
        # If we're initializing from a `deepspeed` CLI command, deepspeed will have already been initialized, as
        # indicated by the presence of the LOCAL_RANK var. Otherwise, we're initializing from Ray / torchrun, and will
        # need to set this var ourselves, then init DeepSpeed here.
        local_rank, local_size = os.environ.get("LOCAL_RANK"), os.environ.get("LOCAL_SIZE")
        init_deepspeed = local_rank is None or local_size is None

        super().__init__(**kwargs)
        self.zero_optimization = zero_optimization or DEFAULT_ZERO_OPTIMIZATION

        if init_deepspeed:
            os.environ["LOCAL_RANK"] = str(self.local_rank())
            os.environ["LOCAL_SIZE"] = str(self.local_size())
            deepspeed.init_distributed()

    def _log_on_init(self):
        logging.info("Using DeepSpeed strategy")

    def prepare(
        self, model: nn.Module, optimizer: Optimizer, lr_scheduler: "LRScheduler", trainer_config: "ECDTrainerConfig"
    ) -> Tuple[nn.Module, Optimizer, "LRScheduler"]:
        # If `batch_size=auto`, we set to 2 temporarily until auto-tuning adjusts it`
        batch_size = trainer_config.batch_size if isinstance(trainer_config.batch_size, int) else 2
        ds_config = {
            "amp": {
                "enabled": trainer_config.use_mixed_precision,
            },
            "zero_optimization": self.zero_optimization,
            "gradient_clipping": trainer_config.gradient_clipping.clipglobalnorm,
            "train_batch_size": batch_size * self.size(),
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
            "steps_per_print": trainer_config.steps_per_checkpoint or 10000,
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=None,  # Don't let DeepSpeed manage the learning rate scheduler
            config=ds_config,
            dist_init_required=False,
        )
        return model_engine, optimizer, lr_scheduler

    def to_device(self, model: nn.Module) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor, model: nn.Module):
        # See: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py
        # runs backpropagation and handles mixed precision
        model.backward(loss)

        # Deepspeed's `engine.step` performs the following operations:
        # - gradient accumulation check
        # - gradient clipping
        # - optimizer step
        # - zero grad
        # - checking overflow
        # - lr_scheduler step (only if engine.lr_scheduler is not None)
        model.step()
        # and this plugin overrides the above calls with no-ops when Accelerate runs under
        # Deepspeed, but allows normal functionality for non-Deepspeed cases thus enabling a simple
        # training loop that works transparently under many training regimes.

    def step(self, optimizer: Optimizer, *args, **kwargs):
        # Handled by `self.backward(loss)`
        pass

    def zero_grad(self, optimizer: Optimizer):
        # Handled by `self.backward(loss)`
        pass

    def set_batch_size(self, model: nn.Module, batch_size: int):
        # Adapted from:
        # https://github.com/microsoft/DeepSpeed/blob/7ce371b139521b1ebbf052f0496b1a16397c1d19/deepspeed/runtime/engine.py#L422  # noqa: E501
        model._config.micro_batch_size_per_gpu = batch_size
        model._config.train_batch_size = batch_size * self.size() * model._config.gradient_accumulation_steps

    def barrier(self):
        deepspeed.comm.barrier()

    def allow_gradient_accumulation(self) -> bool:
        """DeepSpeed handles gradient accumulation internally."""
        return False

    def allow_mixed_precision(self) -> bool:
        """DeepSpeed handles mixed precision internally."""
        return False

    def allow_clip_gradients(self) -> bool:
        """DeepSpeed handles gradient clipping internally."""
        return False

    def prepare_before_load(self) -> bool:
        """DeepSpeed requires the engine to be re-initialized before loading.

        https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html#loading-training-checkpoints
        """
        return True

    @classmethod
    def is_model_parallel(cls) -> bool:
        return True

    def eval(self, model: nn.Module):
        # TODO(travis): remove this when DeepSpeed resolves issue:
        # https://github.com/microsoft/DeepSpeed/issues/3068
        pass

    def create_checkpoint_handle(
        self,
        dist_model: nn.Module,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional["LRScheduler"] = None,
    ) -> Checkpoint:
        return DeepSpeedCheckpoint(self, dist_model, optimizer, scheduler)


class DeepSpeedCheckpoint(Checkpoint):
    def prepare(self, directory: str):
        if self.distributed.local_rank() == 0:
            # Checkpoints need to be written on every rank, but the directory only needs to be created once per node.
            super().prepare(directory)

    def load(self, save_path: str, device: Optional[torch.device] = None) -> bool:
        _, client_state = self.model.load_checkpoint(save_path, load_lr_scheduler_states=False)
        self.global_step = self._get_global_step(client_state, save_path)
        if self.scheduler is not None and "scheduler_state" in client_state:
            self.scheduler.load_state_dict(client_state["scheduler_state"])
        return True

    def save(self, save_path: str, global_step: int):
        client_state = {
            "global_step": global_step,
        }
        if self.scheduler is not None:
            client_state["scheduler_state"] = self.scheduler.state_dict()

        self.model.save_checkpoint(save_path, client_state=client_state)

    def get_state_for_inference(self, save_path: str, device: Optional[torch.device] = None) -> Mapping[str, Any]:
        return get_fp32_state_dict_from_zero_checkpoint(save_path)
