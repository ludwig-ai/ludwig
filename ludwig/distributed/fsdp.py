import logging
from typing import Tuple

from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer

from ludwig.distributed.ddp import DDPStrategy
from ludwig.modules.lr_scheduler import LRScheduler
from ludwig.schema.trainer import ECDTrainerConfig


class FSDPStrategy(DDPStrategy):
    def _log_on_init(self):
        logging.info("Using FSDP strategy")

    def prepare(
        self, model: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler, trainer_config: ECDTrainerConfig
    ) -> Tuple[nn.Module, Optimizer, LRScheduler]:
        return FSDP(model), optimizer, lr_scheduler

    def to_device(self, model: nn.Module) -> nn.Module:
        return model
