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

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import psutil
import torch

from ludwig.backend.utils.storage import StorageManager
from ludwig.data.cache.manager import CacheManager
from ludwig.data.dataframe.pandas import PANDAS
from ludwig.data.dataset.base import DatasetManager
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.models.base import BaseModel
from ludwig.schema.trainer import ECDTrainerConfig, GBMTrainerConfig
from ludwig.utils.fs_utils import get_bytes_obj_from_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import initialize_pytorch
from ludwig.utils.types import Series


class Backend(ABC):
    def __init__(
        self,
        dataset_manager: DatasetManager,
        cache_dir: Optional[str] = None,
        credentials: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        credentials = credentials or {}
        self._dataset_manager = dataset_manager
        self._storage_manager = StorageManager(**credentials)
        self._cache_manager = CacheManager(self._dataset_manager, cache_dir)

    @property
    def storage(self) -> StorageManager:
        return self._storage_manager

    @property
    def cache(self) -> CacheManager:
        return self._cache_manager

    @property
    def dataset_manager(self) -> DatasetManager:
        return self._dataset_manager

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def initialize_pytorch(self, *args, **kwargs):
        raise NotImplementedError()

    @contextmanager
    @abstractmethod
    def create_trainer(self, **kwargs) -> "BaseTrainer":  # noqa: F821
        raise NotImplementedError()

    @abstractmethod
    def sync_model(self, model):
        raise NotImplementedError()

    @abstractmethod
    def broadcast_return(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def is_coordinator(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def df_engine(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def supports_multiprocessing(self):
        raise NotImplementedError()

    @abstractmethod
    def check_lazy_load_supported(self, feature):
        raise NotImplementedError()

    @abstractmethod
    def read_binary_files(self, column: Series, map_fn: Optional[Callable] = None) -> Series:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_available_resources(self) -> Resources:
        raise NotImplementedError()

    @abstractmethod
    def max_concurrent_trials(self, hyperopt_config: Dict[str, Any]) -> Union[int, None]:
        raise NotImplementedError()


class LocalPreprocessingMixin:
    @property
    def df_engine(self):
        return PANDAS

    @property
    def supports_multiprocessing(self):
        return True

    def check_lazy_load_supported(self, feature):
        pass

    def read_binary_files(
        self, column: pd.Series, map_fn: Optional[Callable] = None, file_size: Optional[int] = None
    ) -> pd.Series:
        column = column.fillna(np.nan).replace([np.nan], [None])  # normalize NaNs to None

        sample_fname = column.head(1).values[0]
        with ThreadPoolExecutor() as executor:  # number of threads is inferred
            if isinstance(sample_fname, str):
                result = executor.map(
                    lambda path: get_bytes_obj_from_path(path) if path is not None else path, column.values
                )
            else:
                # If the sample path is not a string, assume the paths has already been read in
                result = column.values

            if map_fn is not None:
                result = executor.map(map_fn, result)

        return pd.Series(result, index=column.index, name=column.name)


class LocalTrainingMixin:
    def initialize_pytorch(self, *args, **kwargs):
        initialize_pytorch(*args, **kwargs)

    def create_trainer(
        self, config: Union[ECDTrainerConfig, GBMTrainerConfig], model: BaseModel, **kwargs
    ) -> "BaseTrainer":  # noqa: F821
        from ludwig.trainers.registry import trainers_registry

        trainers_for_model = get_from_registry(model.type(), trainers_registry)

        trainer_cls = get_from_registry(config.type, trainers_for_model)

        return trainer_cls(config=config, model=model, **kwargs)

    def create_predictor(self, model: BaseModel, **kwargs):
        from ludwig.models.predictor import Predictor

        return Predictor(model, **kwargs)

    def sync_model(self, model):
        pass

    def broadcast_return(self, fn):
        return fn()

    def is_coordinator(self):
        return True


class RemoteTrainingMixin:
    def sync_model(self, model):
        pass

    def broadcast_return(self, fn):
        return fn()

    def is_coordinator(self):
        return True


class LocalBackend(LocalPreprocessingMixin, LocalTrainingMixin, Backend):
    BACKEND_TYPE = "local"

    def __init__(self, **kwargs):
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)

    def initialize(self):
        pass

    @property
    def num_nodes(self) -> int:
        return 1

    def get_available_resources(self) -> Resources:
        return Resources(cpus=psutil.cpu_count(), gpus=torch.cuda.device_count())

    def max_concurrent_trials(self, hyperopt_config: Dict[str, Any]) -> Union[int, None]:
        # Every trial will be run with Pandas and NO Ray Datasets. Allow Ray Tune to use all the
        # trial resources it wants, because there is no Ray Datasets process to compete with it for CPUs.
        return None
