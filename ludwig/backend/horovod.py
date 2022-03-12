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

from ludwig.backend.base import Backend, LocalPreprocessingMixin
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.models.ecd import ECD
from ludwig.models.predictor import Predictor
from ludwig.models.trainer import Trainer
from ludwig.utils.horovod_utils import initialize_horovod
from ludwig.utils.torch_utils import initialize_pytorch


class HorovodBackend(LocalPreprocessingMixin, Backend):
    def __init__(self, **kwargs):
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)
        self._horovod = None

    def initialize(self):
        self._horovod = initialize_horovod()

    def initialize_pytorch(self, *args, **kwargs):
        initialize_pytorch(*args, horovod=self._horovod, **kwargs)

    def create_trainer(self, **kwargs):
        return Trainer(horovod=self._horovod, **kwargs)

    def create_predictor(self, model: ECD, **kwargs):
        return Predictor(model, horovod=self._horovod, **kwargs)

    def sync_model(self, model):
        # Model weights are only saved on the coordinator, so broadcast
        # to all other ranks
        self._horovod.broadcast_parameters(model.state_dict(), root_rank=0)

    def broadcast_return(self, fn):
        """Returns the result of calling `fn` on coordinator, broadcast to all other ranks.

        Specifically, `fn` is only executed on coordinator, but its result is returned by every rank by broadcasting the
        return value from coordinator.
        """
        result = fn() if self.is_coordinator() else None
        if self._horovod:
            name = f"broadcast_return_{int(time.time())}"
            result = self._horovod.broadcast_object(result, name=name)
        return result

    def is_coordinator(self):
        return self._horovod.rank() == 0

    @property
    def num_nodes(self) -> int:
        return self._horovod.size()
