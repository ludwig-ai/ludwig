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

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.utils.storage import StorageManager
from ludwig.constants import MODEL_LLM
from ludwig.data.cache.manager import CacheManager
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.data.dataframe.pandas import PANDAS
from ludwig.data.dataset.base import DatasetManager
from ludwig.data.dataset.pandas import PandasDatasetManager
from ludwig.distributed import init_dist_strategy
from ludwig.distributed.base import DistributedStrategy
from ludwig.models.base import BaseModel
from ludwig.schema.trainer import BaseTrainerConfig
from ludwig.types import HyperoptConfigDict
from ludwig.utils.audio_utils import read_audio_from_path
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator
from ludwig.utils.dataframe_utils import from_batches, to_batches
from ludwig.utils.fs_utils import get_bytes_obj_from_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import initialize_pytorch
from ludwig.utils.types import DataFrame, Series

if TYPE_CHECKING:
    from ludwig.trainers.base import BaseTrainer


@DeveloperAPI
class Backend(ABC):
    def __init__(
        self,
        dataset_manager: DatasetManager,
        cache_dir: str | None = None,
        credentials: dict[str, dict[str, Any]] | None = None,
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
    def create_trainer(self, **kwargs) -> BaseTrainer:
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
    def df_engine(self) -> DataFrameEngine:
        raise NotImplementedError()

    @property
    @abstractmethod
    def supports_multiprocessing(self):
        raise NotImplementedError()

    @abstractmethod
    def read_binary_files(self, column: Series, map_fn: Callable | None = None) -> Series:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def num_training_workers(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_available_resources(self) -> Resources:
        raise NotImplementedError()

    @abstractmethod
    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> int | None:
        raise NotImplementedError()

    @abstractmethod
    def tune_batch_size(self, evaluator_cls: type[BatchSizeEvaluator], dataset_len: int) -> int:
        """Returns best batch size (measured in samples / s) on the given evaluator.

        The evaluator class will need to be instantiated on each worker in the backend cluster, then call
        `evaluator.select_best_batch_size(dataset_len)`.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_transform(self, df: DataFrame, batch_size: int, transform_fn: Callable, name: str = None) -> DataFrame:
        """Applies `transform_fn` to every `batch_size` length batch of `df` and returns the result."""
        raise NotImplementedError()

    def supports_batch_size_tuning(self) -> bool:
        return True


class LocalPreprocessingMixin:
    @property
    def df_engine(self):
        return PANDAS

    @property
    def supports_multiprocessing(self):
        return True

    @staticmethod
    def read_binary_files(column: pd.Series, map_fn: Callable | None = None, file_size: int | None = None) -> pd.Series:
        column = column.fillna(np.nan).replace([np.nan], [None])  # normalize NaNs to None

        sample_fname = column.head(1).values[0]
        with ThreadPoolExecutor() as executor:  # number of threads is inferred
            if isinstance(sample_fname, str):
                if map_fn is read_audio_from_path:  # bypass torchaudio issue that no longer takes in file-like objects
                    result = executor.map(lambda path: map_fn(path) if path is not None else path, column.values)
                else:
                    result = executor.map(
                        lambda path: get_bytes_obj_from_path(path) if path is not None else path, column.values
                    )
            else:
                # If the sample path is not a string, assume the paths has already been read in
                result = column.values

            if map_fn is not None and map_fn is not read_audio_from_path:
                result = executor.map(map_fn, result)

        return pd.Series(result, index=column.index, name=column.name)

    @staticmethod
    def batch_transform(df: DataFrame, batch_size: int, transform_fn: Callable, name: str = None) -> DataFrame:
        name = name or "Batch Transform"
        batches = to_batches(df, batch_size)
        transform = transform_fn()
        out_batches = [transform(batch.reset_index(drop=True)) for batch in tqdm(batches, desc=name)]
        out_df = from_batches(out_batches).reset_index(drop=True)
        return out_df


class LocalTrainingMixin:
    @staticmethod
    def initialize():
        init_dist_strategy("local")

    @staticmethod
    def initialize_pytorch(*args, **kwargs):
        initialize_pytorch(*args, **kwargs)

    def create_trainer(self, config: BaseTrainerConfig, model: BaseModel, **kwargs) -> BaseTrainer:
        from ludwig.trainers.registry import get_llm_trainers_registry, get_trainers_registry

        if model.type() == MODEL_LLM:
            trainer_cls = get_from_registry(config.type, get_llm_trainers_registry())
        else:
            trainer_cls = get_from_registry(model.type(), get_trainers_registry())

        return trainer_cls(config=config, model=model, **kwargs)

    @staticmethod
    def create_predictor(model: BaseModel, **kwargs):
        from ludwig.models.predictor import get_predictor_cls

        return get_predictor_cls(model.type())(model, **kwargs)

    def sync_model(self, model):
        pass

    @staticmethod
    def broadcast_return(fn):
        return fn()

    @staticmethod
    def is_coordinator() -> bool:
        return True

    @staticmethod
    def tune_batch_size(evaluator_cls: type[BatchSizeEvaluator], dataset_len: int) -> int:
        evaluator = evaluator_cls()
        return evaluator.select_best_batch_size(dataset_len)


class RemoteTrainingMixin:
    def sync_model(self, model):
        pass

    @staticmethod
    def broadcast_return(fn):
        return fn()

    @staticmethod
    def is_coordinator() -> bool:
        return True


@DeveloperAPI
class LocalBackend(LocalPreprocessingMixin, LocalTrainingMixin, Backend):
    BACKEND_TYPE = "local"

    @classmethod
    def shared_instance(cls):
        """Returns a shared singleton LocalBackend instance."""
        if not hasattr(cls, "_shared_instance"):
            cls._shared_instance = cls()
        return cls._shared_instance

    def __init__(self, **kwargs):
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)

    @property
    def num_nodes(self) -> int:
        return 1

    @property
    def num_training_workers(self) -> int:
        return 1

    def get_available_resources(self) -> Resources:
        return Resources(cpus=psutil.cpu_count(), gpus=torch.cuda.device_count())

    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> int | None:
        # Every trial will be run with Pandas and NO Ray Datasets. Allow Ray Tune to use all the
        # trial resources it wants, because there is no Ray Datasets process to compete with it for CPUs.
        return None


@DeveloperAPI
class DataParallelBackend(LocalPreprocessingMixin, Backend, ABC):
    BACKEND_TYPE = "deepspeed"

    def __init__(self, **kwargs):
        super().__init__(dataset_manager=PandasDatasetManager(self), **kwargs)
        self._distributed: DistributedStrategy | None = None

    @abstractmethod
    def initialize(self):
        pass

    def initialize_pytorch(self, *args, **kwargs):
        initialize_pytorch(
            *args, local_rank=self._distributed.local_rank(), local_size=self._distributed.local_size(), **kwargs
        )

    def create_trainer(self, **kwargs) -> BaseTrainer:
        from ludwig.trainers.trainer import Trainer

        return Trainer(distributed=self._distributed, **kwargs)

    def create_predictor(self, model: BaseModel, **kwargs):
        from ludwig.models.predictor import get_predictor_cls

        return get_predictor_cls(model.type())(model, distributed=self._distributed, **kwargs)

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
        return self._distributed.size() // self._distributed.local_size()

    @property
    def num_training_workers(self) -> int:
        return self._distributed.size()

    def get_available_resources(self) -> Resources:
        # TODO(travis): this double-counts on the same device, it should use a cross-communicator instead
        cpus = torch.as_tensor([psutil.cpu_count()], dtype=torch.int)
        cpus = self._distributed.allreduce(cpus).item()

        gpus = torch.as_tensor([torch.cuda.device_count()], dtype=torch.int)
        gpus = self._distributed.allreduce(gpus).item()

        return Resources(cpus=cpus, gpus=gpus)

    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> int | None:
        # Return None since there is no Ray component
        return None

    def tune_batch_size(self, evaluator_cls: type[BatchSizeEvaluator], dataset_len: int) -> int:
        evaluator = evaluator_cls()
        return evaluator.select_best_batch_size(dataset_len)
