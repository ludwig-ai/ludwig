#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
from typing import Type, Union

import psutil
import torch

from ludwig.backend.base import Backend, LocalPreprocessingMixin
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.distributed.horovod import HorovodStrategy
from ludwig.models.base import BaseModel
from ludwig.models.predictor import Predictor
from ludwig.trainers.trainer import Trainer
from ludwig.types import HyperoptConfigDict
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import initialize_pytorch


class HorovodBackend(LocalPreprocessingMixin, Backend):
    BACKEND_TYPE = "horovod"

    def __init__(self, **kwargs):
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)
        self._distributed = None

    def initialize(self):
        self._distributed = HorovodStrategy()

    def initialize_pytorch(self, *args, **kwargs):
        initialize_pytorch(
            *args, local_rank=self._distributed.local_rank(), local_size=self._distributed.local_size(), **kwargs
        )

    def create_trainer(self, **kwargs) -> "BaseTrainer":  # noqa: F821
        return Trainer(distributed=self._distributed, **kwargs)

    def create_predictor(self, model: BaseModel, **kwargs):
        return Predictor(model, distributed=self._distributed, **kwargs)

    def sync_model(self, model):
        # Model weights are only saved on the coordinator, so broadcast
        # to all other ranks
        self._distributed.sync_model(model)

    def broadcast_return(self, fn):
        """Returns the result of calling `fn` on coordinator, broadcast to all other ranks.

        Specifically, `fn` is only executed on coordinator, but its result is returned by every rank by broadcasting the
        return value from coordinator.
        """
        result = fn() if self.is_coordinator() else None
        if self._distributed:
            name = f"broadcast_return_{int(time.time())}"
            result = self._distributed.broadcast_object(result, name=name)
        return result

    def is_coordinator(self):
        return self._distributed.rank() == 0

    @property
    def num_nodes(self) -> int:
        return self._distributed.size()

    def get_available_resources(self) -> Resources:
        # TODO(travis): this double-counts on the same device, it should use a cross-communicator instead
        cpus = torch.as_tensor([psutil.cpu_count()], dtype=torch.int)
        cpus = self._distributed.allreduce(cpus).item()

        gpus = torch.as_tensor([torch.cuda.device_count()], dtype=torch.int)
        gpus = self._distributed.allreduce(gpus).item()

        return Resources(cpus=cpus, gpus=gpus)

    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> Union[int, None]:
        # Return None since there is no Ray component
        return None

    def tune_batch_size(self, evaluator_cls: Type[BatchSizeEvaluator], dataset_len: int) -> int:
        evaluator = evaluator_cls()
        return evaluator.select_best_batch_size(dataset_len)
