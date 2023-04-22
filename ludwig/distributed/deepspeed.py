import logging
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import deepspeed
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from ludwig.distributed.ddp import DDPStrategy

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
        super().__init__(**kwargs)
        self.zero_optimization = zero_optimization or DEFAULT_ZERO_OPTIMIZATION

    def _log_on_init(self):
        logging.info("Using DeepSpeed strategy")

    def prepare(
        self, model: nn.Module, optimizer: Optimizer, lr_scheduler: "LRScheduler", trainer_config: "ECDTrainerConfig"
    ) -> Tuple[nn.Module, Optimizer, "LRScheduler"]:
        ds_config = {
            "amp": {
                "enabled": trainer_config.use_mixed_precision,
            },
            "zero_optimization": self.zero_optimization,
            "gradient_clipping": trainer_config.gradient_clipping.clipglobalnorm,
            "train_batch_size": trainer_config.batch_size * self.size(),
            "train_micro_batch_size_per_gpu": trainer_config.batch_size,
            "gradient_accumulation_steps": trainer_config.gradient_accumulation_steps,
            "steps_per_print": trainer_config.steps_per_checkpoint or 10000,
        }
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=None,  # Don't let DeepSpeed manager the learning rate scheduler
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

    def allow_gradient_accumulation(self) -> bool:
        """DeepSpeed handles gradient accumulation internally."""
        return False

    def allow_mixed_precision(self) -> bool:
        """DeepSpeed handles mixed precision internally."""
        return False

    def allow_clip_gradients(self) -> bool:
        """DeepSpeed handles gradient clipping internally."""
        return False
