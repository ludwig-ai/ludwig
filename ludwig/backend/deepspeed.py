import deepspeed

from ludwig.backend.base import DataParallelBackend
from ludwig.distributed import init_dist_strategy


class DeepSpeedBackend(DataParallelBackend):
    BACKEND_TYPE = "deepspeed"

    def initialize(self):
        deepspeed.init_distributed()
        self._distributed = init_dist_strategy("deepspeed")
