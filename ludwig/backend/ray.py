#! /usr/bin/env python
# coding=utf-8
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

import logging
from collections import defaultdict

import ray
from horovod.ray import RayExecutor

from ludwig.backend.base import Backend
from ludwig.constants import NAME
from ludwig.data.processor.dask import DaskProcessor
from ludwig.models.trainer import BaseTrainer, Trainer
from ludwig.utils.tf_utils import initialize_tensorflow


logger = logging.getLogger(__name__)


def get_horovod_kwargs():
    resources = [node['Resources'] for node in ray.state.nodes()]
    use_gpu = int(ray.cluster_resources().get('GPU', 0)) > 0

    # Our goal is to maximize the number of training resources we can
    # form into a homogenous configuration. The priority is GPUs, but
    # can fall back to CPUs if there are no GPUs available.
    key = 'GPU' if use_gpu else 'CPU'

    # Bucket the per node resources by the number of the target resource
    # available on that host (equivalent to number of slots).
    buckets = defaultdict(list)
    for node_resources in resources:
        buckets[int(node_resources.get(key, 0))].append(node_resources)

    # Maximize for the total number of the target resource = num_slots * num_workers
    def get_total_resources(bucket):
        slots, resources = bucket
        return slots * len(resources)

    best_slots, best_resources = max(buckets.items(), key=get_total_resources)
    return dict(
        num_slots=best_slots,
        num_hosts=len(best_resources),
        use_gpu=use_gpu
    )


class RayTrainer(BaseTrainer):
    def __init__(self, horovod_kwargs, trainer_kwargs):
        setting = RayExecutor.create_settings(timeout_s=30)
        self.executor = RayExecutor(setting, **{**get_horovod_kwargs(), **horovod_kwargs})
        self.executor.start(executable_cls=Trainer, executable_kwargs=trainer_kwargs)

    def train(self, *args, **kwargs):
        results = self.executor.execute(lambda trainer: trainer.train(*args, **kwargs))
        return results[0]

    def train_online(self, *args, **kwargs):
        results = self.executor.execute(lambda trainer: trainer.train_online(*args, **kwargs))
        return results[0]

    @property
    def validation_field(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_field)

    @property
    def validation_metric(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_metric)

    def shutdown(self):
        self.executor.shutdown()


class RayBackend(Backend):
    def __init__(self, horovod_kwargs=None):
        super().__init__()
        self._processor = DaskProcessor()
        self._horovod_kwargs = horovod_kwargs or {}

    def initialize(self):
        try:
            ray.init('auto', ignore_reinit_error=True)
        except ConnectionError:
            logger.info('Initializing new Ray cluster...')
            ray.init()

    def initialize_tensorflow(self, *args, **kwargs):
        # Make sure we don't claim any GPU resources on the head node
        initialize_tensorflow(gpus=-1)

    def create_trainer(self, **kwargs):
        return RayTrainer(self._horovod_kwargs, kwargs)

    @property
    def processor(self):
        return self._processor

    @property
    def supports_multiprocessing(self):
        return False

    def check_lazy_load_supported(self, feature):
        raise ValueError(f'RayBackend does not support lazy loading of data files at train time. '
                         f'Set preprocessing config `in_memory: True` for feature {feature[NAME]}')
