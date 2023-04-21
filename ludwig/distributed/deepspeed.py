import logging
from typing import Tuple

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine
from torch import nn
import torch
from torch.optim.optimizer import Optimizer

from ludwig.distributed.ddp import DDPStrategy
from ludwig.modules.lr_scheduler import LRScheduler
from ludwig.schema.trainer import ECDTrainerConfig


class DeepSpeedStrategy(DDPStrategy):
    def _log_on_init(self):
        logging.info("Using DeepSpeed strategy")

    def prepare(
        self, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, trainer_config: ECDTrainerConfig
    ) -> Tuple[nn.Module, Optimizer, LRScheduler]:
        ds_config = {
            "bf16": {"enabled": "auto"},
            "amp": {
                "enabled": trainer_config.use_mixed_precision,
            },
            "zero_optimization": {
                "stage": "auto",
                "stage3_gather_16bit_weights_on_model_save": "auto",
                "offload_optimizer": {"device": "auto"},
                "offload_param": {"device": "auto"},
            },
            "gradient_clipping": trainer_config.gradient_clipping_config.clipglobalnorm,
            "train_batch_size": trainer_config.batch_size * self.size(),
            "train_micro_batch_size_per_gpu": trainer_config.batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
            "steps_per_print": trainer_config.steps_per_checkpoint or 10000,
        }
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
            dist_init_required=False,
        )
        return model_engine, optimizer, lr_scheduler

    def to_device(self, model: nn.Module) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor, model: nn.Module):
        model.backward(loss)


# Helpers taken from Accelerate: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/deepspeed.py
class DeepSpeedEngineWrapper:
    """
    Internal wrapper for deepspeed.runtime.engine.DeepSpeedEngine. This is used to follow conventional training loop.
    Args:
        engine (deepspeed.runtime.engine.DeepSpeedEngine): deepspeed engine to wrap
    """

    def __init__(self, engine: DeepSpeedEngine):
        self.engine = engine

    def backward(self, loss, **kwargs):
        # runs backpropagation and handles mixed precision
        self.engine.backward(loss, **kwargs)

        # Deepspeed's `engine.step` performs the following operations:
        # - gradient accumulation check
        # - gradient clipping
        # - optimizer step
        # - zero grad
        # - checking overflow
        # - lr_scheduler step (only if engine.lr_scheduler is not None)
        self.engine.step()
        # and this plugin overrides the above calls with no-ops when Accelerate runs under
        # Deepspeed, but allows normal functionality for non-Deepspeed cases thus enabling a simple
        # training loop that works transparently under many training regimes.


class DeepSpeedOptimizerWrapper(Optimizer):
    """
    Internal wrapper around a deepspeed optimizer.
    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer):
        super().__init__(optimizer, device_placement=False, scaler=None)
        self.__has_overflow__ = hasattr(self.optimizer, "overflow")

    def zero_grad(self, set_to_none=None):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed

    def step(self):
        pass  # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed

    @property
    def step_was_skipped(self):
        """Whether or not the optimizer step was done, or skipped because of gradient overflow."""
        if self.__has_overflow__:
            return self.optimizer.overflow
        return False
