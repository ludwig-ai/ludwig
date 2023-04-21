from ludwig.backend.base import DataParallelBackend
from ludwig.distributed import init_dist_strategy


class DeepSpeedBackend(DataParallelBackend):
    BACKEND_TYPE = "deepspeed"

    def initialize(self):
        self._distributed = init_dist_strategy("deepspeed")
