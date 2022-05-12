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

import contextlib
import copy
import logging
from distutils.version import LooseVersion
from functools import partial
from typing import Any, Dict, List, Optional

import dask
import numpy as np
import pandas as pd
import ray
import torch
from ray import ObjectRef
from ray.data.dataset_pipeline import DatasetPipeline
from ray.data.extensions import TensorDtype
from ray.util.dask import ray_dask_get

from ludwig.backend.base import Backend, RemoteTrainingMixin
from ludwig.constants import NAME, PREPROCESSING, PROC_COLUMN
from ludwig.data.dataset.ray import RayDataset, RayDatasetManager, RayDatasetShard
from ludwig.models.ecd import ECD
from ludwig.models.predictor import BasePredictor, get_output_columns, Predictor, RemotePredictor
from ludwig.models.trainer import BaseTrainer, RemoteTrainer, TrainerConfig
from ludwig.utils.horovod_utils import initialize_horovod
from ludwig.utils.torch_utils import initialize_pytorch

_ray112 = LooseVersion(ray.__version__) >= LooseVersion("1.12")
import ray.train as rt  # noqa: E402
from ray.train.trainer import Trainer  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from horovod.ray import RayExecutor
except ImportError as e:
    logger.warn(f"ImportError (ray.py) from horovod.ray import RayExecutor failed with error: \n\t{e}")
    RayExecutor = None

if _ray112:
    from ludwig.backend._ray112_compat import HorovodConfig
else:
    from ray.train.horovod import HorovodConfig


# TODO: deprecated v0.5
def get_horovod_kwargs(use_gpu=None):
    # Our goal is to have a worker per resource used for training.
    # The priority is GPUs, but can fall back to CPUs if there are no
    # GPUs available.
    if use_gpu is None:
        use_gpu = int(ray.cluster_resources().get("GPU", 0)) > 0

    resource = "GPU" if use_gpu else "CPU"
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
        use_gpu = int(ray.cluster_resources().get("GPU", 0)) > 0

    if use_gpu:
        num_workers = int(ray.cluster_resources().get("GPU", 0))
    else:
        # TODO: use placement groups or otherwise spread across nodes
        node_resources = [node["Resources"] for node in ray.state.nodes()]
        num_workers = len(node_resources)

    return dict(
        # TODO travis: replace backend here once ray 1.8 released
        # backend='horovod',
        backend=HorovodConfig(),
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={
            "CPU": 0 if use_gpu else 1,
            "GPU": 1 if use_gpu else 0,
        },
    )


def _create_dask_engine(**kwargs):
    from ludwig.data.dataframe.dask import DaskEngine

    return DaskEngine(**kwargs)


def _create_modin_engine(**kwargs):
    from ludwig.data.dataframe.modin import ModinEngine

    return ModinEngine(**kwargs)


def _create_pandas_engine(**kwargs):
    from ludwig.data.dataframe.pandas import PandasEngine

    return PandasEngine(**kwargs)


_engine_registry = {
    "dask": _create_dask_engine,
    "modin": _create_modin_engine,
    "pandas": _create_pandas_engine,
}


def _get_df_engine(processor):
    logger.info(f"Ray processor params: {processor}")
    if processor is None:
        # TODO ray: find an informed way to set the parallelism, in practice
        #  it looks like Dask handles this well on its own most of the time
        return _create_dask_engine()

    processor_kwargs = processor.copy()

    dtype = processor_kwargs.pop("type", "dask")
    engine_cls = _engine_registry.get(dtype)

    return engine_cls(**processor_kwargs)


def train_fn(
    executable_kwargs: Dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: Dict[str, Any] = None,
    features: Dict[str, Dict] = None,
    **kwargs,
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    try:
        initialize_pytorch(horovod=hvd)

        train_shard = RayDatasetShard(
            rt.get_dataset_shard("train"),
            features,
            training_set_metadata,
        )

        try:
            val_shard = rt.get_dataset_shard("val")
        except KeyError:
            val_shard = None

        if val_shard is not None:
            val_shard = RayDatasetShard(
                val_shard,
                features,
                training_set_metadata,
            )

        try:
            test_shard = rt.get_dataset_shard("test")
        except KeyError:
            test_shard = None

        if test_shard is not None:
            test_shard = RayDatasetShard(
                test_shard,
                features,
                training_set_metadata,
            )

        model = ray.get(model_ref)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        trainer = RemoteTrainer(model=model, horovod=hvd, **executable_kwargs)
        results = trainer.train(train_shard, val_shard, test_shard, **kwargs)

        if results is not None:
            # only return the model state dict back to the head node.
            trained_model, *args = results
            results = (trained_model.cpu().state_dict(), *args)

        torch.cuda.empty_cache()

        train_results = results, trainer.validation_field, trainer.validation_metric

    finally:
        hvd.shutdown()
    return train_results


@ray.remote
def tune_batch_size_fn(
    dataset: RayDataset = None,
    data_loader_kwargs: Dict[str, Any] = None,
    executable_kwargs: Dict[str, Any] = None,
    model: ECD = None,  # noqa: F821
    ludwig_config: Dict[str, Any] = None,
    training_set_metadata: Dict[str, Any] = None,
    features: Dict[str, Dict] = None,
    **kwargs,
) -> int:
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    try:
        initialize_pytorch(horovod=hvd)

        pipe = dataset.pipeline(shuffle=False, **data_loader_kwargs)
        train_shard = RayDatasetShard(
            pipe,
            features,
            training_set_metadata,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        trainer = RemoteTrainer(model=model, horovod=hvd, **executable_kwargs)
        return trainer.tune_batch_size(ludwig_config, train_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()
        hvd.shutdown()


@ray.remote
def tune_learning_rate_fn(
    dataset: RayDataset,
    config: Dict[str, Any],
    data_loader_kwargs: Dict[str, Any] = None,
    executable_kwargs: Dict[str, Any] = None,
    model: ECD = None,  # noqa: F821
    training_set_metadata: Dict[str, Any] = None,
    features: Dict[str, Dict] = None,
    **kwargs,
) -> float:
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    try:
        initialize_pytorch(horovod=hvd)

        pipe = dataset.pipeline(shuffle=False, **data_loader_kwargs)
        train_shard = RayDatasetShard(
            pipe,
            features,
            training_set_metadata,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        trainer = RemoteTrainer(model=model, horovod=hvd, **executable_kwargs)
        return trainer.tune_learning_rate(config, train_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()
        hvd.shutdown()


class RayTrainerV2(BaseTrainer):
    def __init__(self, model, trainer_kwargs, data_loader_kwargs, executable_kwargs):
        self.model = model.cpu()
        self.data_loader_kwargs = data_loader_kwargs
        self.executable_kwargs = executable_kwargs
        self.trainer_kwargs = trainer_kwargs
        self._validation_field = None
        self._validation_metric = None

    @contextlib.contextmanager
    def create_runner(self):
        trainer = Trainer(**{**get_trainer_kwargs(), **self.trainer_kwargs})
        trainer.start()
        try:
            yield trainer
        finally:
            trainer.shutdown()

    def train(
        self,
        training_set: RayDataset,
        validation_set: Optional[RayDataset] = None,
        test_set: Optional[RayDataset] = None,
        **kwargs,
    ):
        executable_kwargs = self.executable_kwargs

        kwargs = {
            "training_set_metadata": training_set.training_set_metadata,
            "features": training_set.features,
            **kwargs,
        }

        dataset = {"train": training_set.pipeline(**self.data_loader_kwargs)}
        if validation_set is not None:
            dataset["val"] = validation_set.pipeline(shuffle=False, **self.data_loader_kwargs)
        if test_set is not None:
            dataset["test"] = test_set.pipeline(shuffle=False, **self.data_loader_kwargs)

        with self.create_runner() as runner:
            results, self._validation_field, self._validation_metric = runner.run(
                lambda config: train_fn(**config),
                config={"executable_kwargs": executable_kwargs, "model_ref": ray.put(self.model), **kwargs},
                dataset=dataset,
            )[0]

        # load state dict back into the model
        state_dict, *args = results
        self.model.load_state_dict(state_dict)
        results = (self.model, *args)

        return results

    def train_online(self, *args, **kwargs):
        raise NotImplementedError()

    def tune_batch_size(
        self,
        config: Dict[str, Any],
        training_set: RayDataset,
        **kwargs,
    ) -> int:
        return ray.get(
            tune_batch_size_fn.options(num_cpus=self.num_cpus, num_gpus=self.num_gpus).remote(
                dataset=training_set,
                data_loader_kwargs=self.data_loader_kwargs,
                executable_kwargs=self.executable_kwargs,
                model=ray.put(self.model),
                ludwig_config=config,
                training_set_metadata=training_set.training_set_metadata,
                features=training_set.features,
                **kwargs,
            )
        )

    def tune_learning_rate(self, config, training_set: RayDataset, **kwargs) -> float:
        return ray.get(
            tune_learning_rate_fn.options(num_cpus=self.num_cpus, num_gpus=self.num_gpus).remote(
                dataset=training_set,
                config=config,
                data_loader_kwargs=self.data_loader_kwargs,
                executable_kwargs=self.executable_kwargs,
                model=ray.put(self.model),
                training_set_metadata=training_set.training_set_metadata,
                features=training_set.features,
                **kwargs,
            )
        )

    @property
    def validation_field(self):
        return self._validation_field

    @property
    def validation_metric(self):
        return self._validation_metric

    @property
    def config(self) -> TrainerConfig:
        return self.executable_kwargs["config"]

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.config.batch_size = value

    @property
    def eval_batch_size(self) -> int:
        return self.config.eval_batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, value: int):
        self.config.eval_batch_size = value

    @property
    def resources_per_worker(self) -> Dict[str, Any]:
        trainer_kwargs = {**get_trainer_kwargs(), **self.trainer_kwargs}
        return trainer_kwargs.get("resources_per_worker", {})

    @property
    def num_cpus(self) -> int:
        return self.resources_per_worker.get("CPU", 1)

    @property
    def num_gpus(self) -> int:
        return self.resources_per_worker.get("GPU", 0)

    def set_base_learning_rate(self, learning_rate: float):
        self.config.learning_rate = learning_rate

    def shutdown(self):
        pass


def legacy_train_fn(
    trainer: RemoteTrainer = None,
    remote_model: "LudwigModel" = None,  # noqa: F821
    training_set_metadata: Dict[str, Any] = None,
    features: Dict[str, Dict] = None,
    train_shards: List[DatasetPipeline] = None,
    val_shards: List[DatasetPipeline] = None,
    test_shards: List[DatasetPipeline] = None,
    **kwargs,
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    initialize_pytorch(horovod=hvd)

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

    results = trainer.train(train_shard, val_shard, test_shard, **kwargs)
    return results


class HorovodRemoteTrainer(RemoteTrainer):
    def __init__(self, **kwargs):
        horovod = initialize_horovod()
        super().__init__(horovod=horovod, **kwargs)


class RayLegacyTrainer(BaseTrainer):
    def __init__(self, horovod_kwargs, executable_kwargs):
        # TODO ray: make this more configurable by allowing YAML overrides of timeout_s, etc.
        if RayExecutor is None:
            logger.error(
                "RayLegacyTrainer failed to initialize: RayExecutor is None. Make sure horovod[ray] is installed."
            )
            return
        setting = RayExecutor.create_settings(timeout_s=30)

        self.executor = RayExecutor(setting, **{**get_horovod_kwargs(), **horovod_kwargs})
        self.executor.start(executable_cls=HorovodRemoteTrainer, executable_kwargs=executable_kwargs)

    def train(self, model, training_set, validation_set=None, test_set=None, **kwargs):
        workers = self.executor.driver.workers
        train_shards = training_set.pipeline().split(n=len(workers), locality_hints=workers, equal=True)
        val_shards = (
            validation_set.pipeline(shuffle=False).split(n=len(workers), locality_hints=workers)
            if validation_set
            else None
        )
        test_shards = (
            test_set.pipeline(shuffle=False).split(n=len(workers), locality_hints=workers) if test_set else None
        )

        results = self.executor.execute(
            lambda trainer: legacy_train_fn(
                trainer,
                model,
                training_set.training_set_metadata,
                training_set.features,
                train_shards,
                val_shards,
                test_shards,
                **kwargs,
            )
        )

        return results

    def train_online(self, model, *args, **kwargs):
        results = self.executor.execute(lambda trainer: trainer.train_online(model, *args, **kwargs))

        return results[0]

    @property
    def validation_field(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_field)

    @property
    def validation_metric(self):
        return self.executor.execute_single(lambda trainer: trainer.validation_metric)

    def shutdown(self):
        self.executor.shutdown()


def eval_fn(
    predictor_kwargs: Dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: Dict[str, Any] = None,
    features: Dict[str, Dict] = None,
    **kwargs,
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    hvd = initialize_horovod()
    try:
        initialize_pytorch(horovod=hvd)

        eval_shard = RayDatasetShard(
            rt.get_dataset_shard("eval"),
            features,
            training_set_metadata,
        )

        model = ray.get(model_ref)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        predictor = RemotePredictor(model=model, horovod=hvd, **predictor_kwargs)
        return predictor.batch_evaluation(eval_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()
        hvd.shutdown()


class RayPredictor(BasePredictor):
    def __init__(self, model: ECD, trainer_kwargs, data_loader_kwargs, **predictor_kwargs):
        self.batch_size = predictor_kwargs["batch_size"]
        self.trainer_kwargs = trainer_kwargs
        self.data_loader_kwargs = data_loader_kwargs
        self.predictor_kwargs = predictor_kwargs
        self.actor_handles = []
        self.model = model.cpu()

    def batch_predict(self, dataset: RayDataset, *args, **kwargs):
        self._check_dataset(dataset)

        predictor_kwargs = self.predictor_kwargs
        output_columns = get_output_columns(self.model.output_features)
        batch_predictor = self.get_batch_infer_model(
            self.model,
            predictor_kwargs,
            output_columns,
            dataset.features,
            dataset.training_set_metadata,
            *args,
            **kwargs,
        )

        columns = [f.proc_column for f in self.model.input_features.values()]

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = df[c].astype(TensorDtype())
            return df

        trainer_kwargs = {**get_trainer_kwargs(), **self.trainer_kwargs}
        resources_per_worker = trainer_kwargs.get("resources_per_worker", {})
        num_cpus = resources_per_worker.get("CPU", 1)
        num_gpus = resources_per_worker.get("GPU", 0)

        dask_dataset = (
            dataset.ds.map_batches(to_tensors, batch_format="pandas")
            .map_batches(
                batch_predictor,
                batch_size=self.batch_size,
                compute="actors",
                batch_format="pandas",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )
            .to_dask()
        )

        for of_feature in self.model.output_features.values():
            dask_dataset = of_feature.unflatten(dask_dataset)

        return dask_dataset

    def predict_single(self, batch):
        raise NotImplementedError("predict_single can only be called on a local predictor")

    def batch_evaluation(self, dataset: RayDataset, collect_predictions: bool = False, **kwargs):
        # We need to be in a Horovod context to collect the aggregated metrics, since it relies on collective
        # communication ops. However, Horovod is not suitable for transforming one big dataset to another. For that
        # we will use Ray Datasets. Therefore, we break this up into two separate steps, and two passes over the
        # dataset. In the future, we can explore ways to combine these into a single step to reduce IO.
        runner = Trainer(**{**get_trainer_kwargs(), **self.trainer_kwargs})
        runner.start()
        try:
            # Collect eval metrics by distributing work across nodes / gpus with Horovod
            datasets = {"eval": dataset.pipeline(shuffle=False, **self.data_loader_kwargs)}
            predictor_kwargs = {
                **self.predictor_kwargs,
                "collect_predictions": False,
            }
            eval_stats, _ = runner.run(
                lambda config: eval_fn(**config),
                config={
                    "predictor_kwargs": predictor_kwargs,
                    "model_ref": ray.put(self.model),
                    "training_set_metadata": dataset.training_set_metadata,
                    "features": dataset.features,
                    **kwargs,
                },
                dataset=datasets,
            )[0]
        finally:
            runner.shutdown()

        predictions = None
        if collect_predictions:
            # Collect eval predictions by using Ray Datasets to transform partitions of the data in parallel
            predictions = self.batch_predict(dataset)

        return eval_stats, predictions

    def batch_collect_activations(self, model, *args, **kwargs):
        raise NotImplementedError("Ray backend does not support collecting activations at this time.")

    def _check_dataset(self, dataset):
        if not isinstance(dataset, RayDataset):
            raise RuntimeError(f"Ray backend requires RayDataset for inference, " f"found: {type(dataset)}")

    def shutdown(self):
        for handle in self.actor_handles:
            ray.kill(handle)
        self.actor_handles.clear()

    def get_batch_infer_model(
        self,
        model: "LudwigModel",  # noqa: F821
        predictor_kwargs: Dict[str, Any],
        output_columns: List[str],
        features: Dict[str, Dict],
        training_set_metadata: Dict[str, Any],
        *args,
        **kwargs,
    ):
        model_ref = ray.put(model)

        class BatchInferModel:
            def __init__(self):
                model = ray.get(model_ref)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = model.to(device)

                self.output_columns = output_columns
                self.features = features
                self.training_set_metadata = training_set_metadata
                self.reshape_map = {
                    f[PROC_COLUMN]: training_set_metadata[f[NAME]].get("reshape") for f in features.values()
                }
                predictor = Predictor(model, **predictor_kwargs)
                self.predict = partial(predictor.predict_single, *args, **kwargs)

            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                dataset = self._prepare_batch(df)
                predictions = self.predict(batch=dataset)

                for output_feature in self.model.output_features.values():
                    predictions = output_feature.flatten(predictions)
                ordered_predictions = predictions[self.output_columns]
                return ordered_predictions

            def _prepare_batch(self, batch: pd.DataFrame) -> Dict[str, np.ndarray]:
                res = {c: batch[c].to_numpy() for c in self.features.keys()}

                for c in self.features.keys():
                    reshape = self.reshape_map.get(c)
                    if reshape is not None:
                        res[c] = res[c].reshape((-1, *reshape))

                return res

        return BatchInferModel


class RayBackend(RemoteTrainingMixin, Backend):
    def __init__(self, processor=None, trainer=None, loader=None, use_legacy=False, **kwargs):
        super().__init__(dataset_manager=RayDatasetManager(self), **kwargs)
        self._df_engine = _get_df_engine(processor)
        self._horovod_kwargs = trainer or {}
        self._pytorch_kwargs = {}
        self._data_loader_kwargs = loader or {}
        self._use_legacy = use_legacy

    def initialize(self):
        if not ray.is_initialized():
            try:
                ray.init("auto", ignore_reinit_error=True)
            except ConnectionError:
                logger.info("Initializing new Ray cluster...")
                ray.init(ignore_reinit_error=True)

        dask.config.set(scheduler=ray_dask_get)

    def initialize_pytorch(self, **kwargs):
        # Make sure we don't claim any GPU resources on the head node
        initialize_pytorch(gpus=-1)
        self._pytorch_kwargs = kwargs

    def create_trainer(self, model: ECD, **kwargs):
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}
        if not self._use_legacy:
            # Deep copy to workaround https://github.com/ray-project/ray/issues/24139
            return RayTrainerV2(
                model,
                copy.deepcopy(self._horovod_kwargs),
                self._data_loader_kwargs,
                executable_kwargs,
            )
        else:
            # TODO: deprecated 0.5
            return RayLegacyTrainer(self._horovod_kwargs, executable_kwargs)

    def create_predictor(self, model: ECD, **kwargs):
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}
        return RayPredictor(
            model,
            copy.deepcopy(self._horovod_kwargs),
            self._data_loader_kwargs,
            **executable_kwargs,
        )

    def set_distributed_kwargs(self, **kwargs):
        self._horovod_kwargs = kwargs

    @property
    def df_engine(self):
        return self._df_engine

    @property
    def supports_multiprocessing(self):
        return False

    def check_lazy_load_supported(self, feature):
        if not feature[PREPROCESSING]["in_memory"]:
            raise ValueError(
                f"RayBackend does not support lazy loading of data files at train time. "
                f"Set preprocessing config `in_memory: True` for feature {feature[NAME]}"
            )

    @property
    def num_nodes(self) -> int:
        if not ray.is_initialized():
            return 1
        return len(ray.nodes())
