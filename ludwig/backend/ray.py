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
from distutils.version import LooseVersion
from functools import partial
from typing import Any, Dict, List

import dask
import numpy as np
import pandas as pd
import ray
from horovod.ray import RayExecutor
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.extensions import TensorDtype
from ray.util.dask import ray_dask_get

from ludwig.backend.base import Backend, RemoteTrainingMixin
from ludwig.constants import NAME, PREPROCESSING, PROC_COLUMN
from ludwig.data.dataframe.dask import DaskEngine
from ludwig.data.dataframe.pandas import PandasEngine
from ludwig.data.dataset.ray import RayDataset, RayDatasetShard, RayDatasetManager
from ludwig.models.ecd import ECD
from ludwig.models.predictor import BasePredictor, Predictor, get_output_columns
from ludwig.models.trainer import BaseTrainer, RemoteTrainer
from ludwig.utils.horovod_utils import initialize_horovod

# TODO(travis): remove for ray 1.8
from ludwig.utils.torch_utils import initialize_pytorch

_ray18 = LooseVersion(ray.__version__) >= LooseVersion("1.8")
if _ray18:
    import ray.train as rt
    from ray.train.trainer import Trainer, TrainWorkerGroup
    from ray.train.backends.horovod import HorovodConfig
else:
    import ray.util.sgd.v2 as rt
    from ray.util.sgd.v2.trainer import Trainer
    from ray.util.sgd.v2.trainer import SGDWorkerGroup as TrainWorkerGroup
    from ludwig.backend._ray17_compat import HorovodConfig


logger = logging.getLogger(__name__)


# TODO: deprecated v0.5
def get_horovod_kwargs(use_gpu=None):
    # Our goal is to have a worker per resource used for training.
    # The priority is GPUs, but can fall back to CPUs if there are no
    # GPUs available.
    if use_gpu is None:
        use_gpu = int(ray.cluster_resources().get('GPU', 0)) > 0

    resource = 'GPU' if use_gpu else 'CPU'
    num_workers = int(ray.cluster_resources().get(resource, 0))

    return dict(
        num_workers=num_workers,
        use_gpu=use_gpu,
    )


def get_trainer_kwargs(use_gpu=None):
    # Our goal is to have a worker per resource used for training.
    # The priority is GPUs, but can fall back to CPUs if there are no
    # GPUs available.
    if use_gpu is None:
        use_gpu = int(ray.cluster_resources().get('GPU', 0)) > 0

    if use_gpu:
        num_workers = int(ray.cluster_resources().get('GPU', 0))
        # TODO(shreya): Why not assign CPUs per worker?
        resources_per_worker = None
    else:
        # Enforce one worker per node by requesting half the CPUs on the node
        # TODO: use placement groups
        resources = [node['Resources'] for node in ray.state.nodes()]
        min_cpus = min(r['CPU'] for r in resources)
        num_workers = len(resources)
        resources_per_worker = {
            'CPU': min(min_cpus / 2 + 1, min_cpus)
        }

    return dict(
        # TODO travis: replace backend here once ray 1.8 released
        # backend='horovod',
        backend=HorovodConfig(),
        num_workers=num_workers,
        resources_per_worker=resources_per_worker,
        use_gpu=use_gpu,
    )


_engine_registry = {
    'dask': DaskEngine,
    'pandas': PandasEngine,
}


def _get_df_engine(processor):
    logger.info(f"Ray processor params: {processor}")
    if processor is None:
        # TODO ray: find an informed way to set the parallelism, in practice
        #  it looks like Dask handles this well on its own most of the time
        return DaskEngine()

    processor_kwargs = processor.copy()

    dtype = processor_kwargs.pop('type', 'dask')
    engine_cls = _engine_registry.get(dtype)

    return engine_cls(**processor_kwargs)


class RayRemoteTrainer(RemoteTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def train_online(self, *args, **kwargs):
        return super().train_online(*args, **kwargs)


def train_fn(
        executable_kwargs: Dict[str, Any] = None,
        model: "LudwigModel" = None,
        training_set_metadata: Dict[str, Any] = None,
        features: Dict[str, Dict] = None,
        train_shards: List[DatasetPipeline] = None,
        val_shards: List[DatasetPipeline] = None,
        test_shards: List[DatasetPipeline] = None,
        **kwargs
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    initialize_pytorch(horovod=hvd)

    train_shard = RayDatasetShard(
        train_shards[rt.world_rank()], #rt.get_dataset_shard("train"),
        features,
        training_set_metadata,
    )

    val_shard = val_shards[rt.world_rank()] if val_shards else None #rt.get_dataset_shard("val")
    if val_shard is not None:
        val_shard = RayDatasetShard(
            val_shard,
            features,
            training_set_metadata,
        )

    test_shard = test_shards[rt.world_rank()] if test_shards else None #rt.get_dataset_shard("test")
    if test_shard is not None:
        test_shard = RayDatasetShard(
            test_shard,
            features,
            training_set_metadata,
        )

    trainer = RayRemoteTrainer(model=model, **executable_kwargs)
    results = trainer.train(
        train_shard,
        val_shard,
        test_shard,
        **kwargs
    )

    # TODO(shreya): Figure out GPU memory leak
    # Clear CUDA memory, place model on CPU, return model to user
    # torch.cuda.empty_cache()
    # TODO(shreya): Check if placing model off GPU explicitly makes a difference
    # model.cpu()

    return results, trainer.validation_field, trainer.validation_metric


class RayTrainerV2(BaseTrainer):
    def __init__(self, model, trainer_kwargs, executable_kwargs):
        self.model = model
        self.executable_kwargs = executable_kwargs
        self.trainer = Trainer(**{**get_trainer_kwargs(), **trainer_kwargs})
        self.trainer.start()
        self._validation_field = None
        self._validation_metric = None

    def train(self, training_set, validation_set=None, test_set=None, **kwargs):
        executable_kwargs = self.executable_kwargs

        # TODO(travis) remove this once `dataset` keyword is supported:
        wg = TrainWorkerGroup(self.trainer._executor.worker_group)
        workers = [w for w in wg]

        train_shards = training_set.pipeline().split(
            n=len(workers), locality_hints=workers, equal=True
        )
        val_shards = validation_set.pipeline(shuffle=False).split(
            n=len(workers), locality_hints=workers
        ) if validation_set else None
        test_shards = test_set.pipeline(shuffle=False).split(
            n=len(workers), locality_hints=workers
        ) if test_set else None

        kwargs = {
            'train_shards': train_shards,
            'val_shards': val_shards,
            'test_shards': test_shards,
            'training_set_metadata': training_set.training_set_metadata,
            'features': training_set.features,
            **kwargs
        }

        results, self._validation_field, self._validation_metric = self.trainer.run(
            lambda config: train_fn(**config),
            config={
                'executable_kwargs': executable_kwargs,
                'model': self.model,
                **kwargs
            },
            # dataset={
            #     'train': training_set.pipeline(),
            #     'val': validation_set.pipeline() if validation_set else None,
            #     'test': test_set.pipeline() if test_set else None,
            # },
        )[0]

        return results

    def train_online(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def validation_field(self):
        return self._validation_field

    @property
    def validation_metric(self):
        return self._validation_metric

    def shutdown(self):
        self.trainer.shutdown()


def legacy_train_fn(
        trainer: RayRemoteTrainer = None,
        remote_model: "LudwigModel" = None,
        training_set_metadata: Dict[str, Any] = None,
        features: Dict[str, Dict] = None,
        train_shards: List[DatasetPipeline] = None,
        val_shards: List[DatasetPipeline] = None,
        test_shards: List[DatasetPipeline] = None,
        **kwargs
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    initialize_pytorch(horovod=hvd)

    model = remote_model.load()

    train_shard = RayDatasetShard(
        train_shards[hvd.rank()],
        features,
        training_set_metadata,
    )

    val_shard = val_shards[hvd.rank()] if val_shards else None
    if val_shard is not None:
        val_shard = RayDatasetShard(
            val_shard,
            features,
            training_set_metadata,
        )

    test_shard = test_shards[hvd.rank()] if test_shards else None
    if test_shard is not None:
        test_shard = RayDatasetShard(
            test_shard,
            features,
            training_set_metadata,
        )

    results = trainer.train(
        train_shard,
        val_shard,
        test_shard,
        **kwargs
    )
    return results


class RayLegacyTrainer(BaseTrainer):
    def __init__(self, horovod_kwargs, executable_kwargs):
        # TODO ray: make this more configurable by allowing YAML overrides of timeout_s, etc.
        setting = RayExecutor.create_settings(timeout_s=30)

        self.executor = RayExecutor(
            setting, **{**get_horovod_kwargs(), **horovod_kwargs})
        self.executor.start(executable_cls=RayRemoteTrainer,
                            executable_kwargs=executable_kwargs)

    def train(self, model, training_set, validation_set=None, test_set=None, **kwargs):
        workers = self.executor.driver.workers
        train_shards = training_set.pipeline().split(
            n=len(workers), locality_hints=workers, equal=True
        )
        val_shards = validation_set.pipeline(shuffle=False).split(
            n=len(workers), locality_hints=workers
        ) if validation_set else None
        test_shards = test_set.pipeline(shuffle=False).split(
            n=len(workers), locality_hints=workers
        ) if test_set else None

        results = self.executor.execute(
            lambda trainer: legacy_train_fn(
                trainer,
                model,
                training_set.training_set_metadata,
                training_set.features,
                train_shards,
                val_shards,
                test_shards,
                **kwargs)
        )

        return results

    def train_online(self, model, *args, **kwargs):
        results = self.executor.execute(
            lambda trainer: trainer.train_online(
                model, *args, **kwargs)
        )

        return results[0]

    @property
    def validation_field(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_field)

    @property
    def validation_metric(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_metric)

    def shutdown(self):
        self.executor.shutdown()


class RayPredictor(BasePredictor):
    def __init__(self, **predictor_kwargs):
        self.batch_size = predictor_kwargs.get('batch_size', 128)
        self.predictor_kwargs = predictor_kwargs
        self.actor_handles = []

    def batch_predict(self, model: ECD, dataset: RayDataset, *args, **kwargs):
        self._check_dataset(dataset)

        predictor_kwargs = self.predictor_kwargs
        output_columns = get_output_columns(model.output_features)
        batch_predictor = self.get_batch_infer_model(
            model,
            predictor_kwargs,
            output_columns,
            dataset.features,
            dataset.training_set_metadata,
            *args,
            **kwargs
        )

        columns = [
           f.proc_column for f in model.input_features.values()
        ]

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        num_gpus = int(ray.cluster_resources().get('GPU', 0) > 0)
        dask_dataset = dataset.ds.map_batches(
            to_tensors, batch_format='pandas'
        ).map_batches(
            batch_predictor,
            batch_size=self.batch_size,
            compute='actors',
            batch_format='pandas',
            num_gpus=num_gpus
        ).to_dask()

        for of_feature in model.output_features.values():
            dask_dataset = of_feature.unflatten(dask_dataset)

        return dask_dataset

    def predict_single(self, model, batch):
        raise NotImplementedError(
            "predict_single can only be called on a local predictor"
        )

    def batch_evaluation(self, model, dataset, collect_predictions=False, **kwargs):
        raise NotImplementedError(
            'Ray backend does not support batch evaluation at this time.'
        )

    def batch_collect_activations(self, model, *args, **kwargs):
        raise NotImplementedError(
            'Ray backend does not support collecting activations at this time.'
        )

    def _check_dataset(self, dataset):
        if not isinstance(dataset, RayDataset):
            raise RuntimeError(
                f'Ray backend requires RayDataset for inference, '
                f'found: {type(dataset)}'
            )

    def shutdown(self):
        for handle in self.actor_handles:
            ray.kill(handle)
        self.actor_handles.clear()

    def get_batch_infer_model(
            self,
            model: "LudwigModel",
            predictor_kwargs: Dict[str, Any],
            output_columns: List[str],
            features: Dict[str, Dict],
            training_set_metadata: Dict[str, Any],
            *args, **kwargs
    ):
        class BatchInferModel:
            def __init__(self):
                self.model = model
                self.output_columns = output_columns
                self.features = features
                self.training_set_metadata = training_set_metadata
                self.reshape_map = {
                    f[PROC_COLUMN]: training_set_metadata[f[NAME]].get('reshape')
                    for f in features.values()
                }
                predictor = Predictor(**predictor_kwargs)
                self.predict = partial(predictor.predict_single, *args, **kwargs)

            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                dataset = self._prepare_batch(df)
                predictions = self.predict(model=self.model, batch=dataset)

                for output_feature in self.model.output_features.values():
                    predictions = output_feature.flatten(predictions)
                ordered_predictions = predictions[self.output_columns]
                return ordered_predictions

            def _prepare_batch(self, batch: pd.DataFrame) -> Dict[str, np.ndarray]:
                res = {
                    c: batch[c].to_numpy() for c in self.features.keys()
                }

                for c in self.features.keys():
                    reshape = self.reshape_map.get(c)
                    if reshape is not None:
                        res[c] = res[c].reshape((-1, *reshape))

                return res

        return BatchInferModel


class RayBackend(RemoteTrainingMixin, Backend):
    def __init__(self, processor=None, trainer=None, use_legacy=False, **kwargs):
        super().__init__(dataset_manager=RayDatasetManager(self), **kwargs)
        self._df_engine = _get_df_engine(processor)
        self._horovod_kwargs = trainer or {}
        self._pytorch_kwargs = {}
        self._use_legacy = use_legacy

    def initialize(self):
        if not ray.is_initialized():
            try:
                ray.init('auto', ignore_reinit_error=True)
            except ConnectionError:
                logger.info('Initializing new Ray cluster...')
                ray.init(ignore_reinit_error=True)

        dask.config.set(scheduler=ray_dask_get)

    def initialize_pytorch(self, **kwargs):
        # Make sure we don't claim any GPU resources on the head node
        initialize_pytorch(gpus=-1)
        self._pytorch_kwargs = kwargs

    def create_trainer(self, model: ECD, **kwargs):
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}
        if not self._use_legacy:
            return RayTrainerV2(model, self._horovod_kwargs, executable_kwargs)
        else:
            # TODO: deprecated 0.5
            return RayLegacyTrainer(self._horovod_kwargs, executable_kwargs)

    def create_predictor(self, **kwargs):
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}
        return RayPredictor(**executable_kwargs)

    def set_distributed_kwargs(self, **kwargs):
        self._horovod_kwargs = kwargs

    @property
    def df_engine(self):
        return self._df_engine

    @property
    def supports_multiprocessing(self):
        return False

    def check_lazy_load_supported(self, feature):
        if not feature[PREPROCESSING]['in_memory']:
            raise ValueError(f'RayBackend does not support lazy loading of data files at train time. '
                             f'Set preprocessing config `in_memory: True` for feature {feature[NAME]}')
