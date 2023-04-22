from typing import Any, Dict, Optional
import deepspeed

from ludwig.backend.base import DataParallelBackend
from ludwig.distributed import init_dist_strategy


class DeepSpeedBackend(DataParallelBackend):
    BACKEND_TYPE = "deepspeed"

    def __init__(self, zero_optimization: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.zero_optimization = zero_optimization

    def initialize(self):
        deepspeed.init_distributed()
        self._distributed = init_dist_strategy("deepspeed", zero_optimization=self.zero_optimization)
