import logging

from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ludwig.distributed.ddp import DDPStrategy


class FSDPStrategy(DDPStrategy):
    def _log_on_init(self):
        logging.info("Using FSDP strategy")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        return FSDP(model)
