import logging
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from ludwig.distributed.ddp import DDPStrategy
from ludwig.modules.optimization_modules import create_optimizer

if TYPE_CHECKING:
    from ludwig.schema.trainer import ECDTrainerConfig


class FSDPStrategy(DDPStrategy):
    def _log_on_init(self):
        logging.info("Using FSDP strategy")

    def prepare(
        self,
        model: nn.Module,
        trainer_config: "ECDTrainerConfig",
        base_learning_rate: float,
    ) -> Tuple[nn.Module, Optimizer]:
        return FSDP(model), create_optimizer(model, trainer_config.optimizer, base_learning_rate)

    def to_device(self, model: nn.Module, device: Optional[torch.device] = None) -> nn.Module:
        return model

    @classmethod
    def is_model_parallel(cls) -> bool:
        return True
