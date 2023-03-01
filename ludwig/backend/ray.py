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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

import dask
import numpy as np
import pandas as pd
import ray
import torch
import tqdm
from packaging import version
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray import ObjectRef
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import DatasetConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchCheckpoint
from ray.util.dask import ray_dask_get
from ray.util.placement_group import placement_group, remove_placement_group

from ludwig.distributed import get_current_dist_strategy, get_default_strategy_name, get_dist_strategy
from ludwig.utils.batch_size_tuner import BatchSizeEvaluator

if TYPE_CHECKING:
    from ludwig.api import LudwigModel

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.base import Backend, RemoteTrainingMixin
from ludwig.backend.datasource import BinaryIgnoreNoneTypeDatasource
from ludwig.constants import CPU_RESOURCES_PER_TRIAL, EXECUTOR, MODEL_ECD, NAME, PROC_COLUMN, TYPE
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.data.dataframe.dask import tensor_extension_casting
from ludwig.data.dataset.ray import _SCALAR_TYPES, RayDataset, RayDatasetManager, RayDatasetShard
from ludwig.models.base import BaseModel
from ludwig.models.ecd import ECD
from ludwig.models.predictor import BasePredictor, get_output_columns, Predictor, RemotePredictor
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.trainers.registry import ray_trainers_registry, register_ray_trainer
from ludwig.trainers.trainer import BaseTrainer, RemoteTrainer
from ludwig.types import HyperoptConfigDict, ModelConfigDict, TrainerConfigDict, TrainingSetMetadataDict
from ludwig.utils.dataframe_utils import set_index_name
from ludwig.utils.fs_utils import get_fs_and_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import get_torch_device, initialize_pytorch
from ludwig.utils.types import DataFrame, Series

_ray230 = version.parse(ray.__version__) >= version.parse("2.3.0")


logger = logging.getLogger(__name__)


RAY_DEFAULT_PARALLELISM = 200
FIFTEEN_MINS_IN_S = 15 * 60


def _num_nodes() -> int:
    node_resources = [node["Resources"] for node in ray.nodes()]
    return len(node_resources)


def get_trainer_kwargs(**kwargs) -> TrainerConfigDict:
    kwargs = copy.deepcopy(kwargs)

    # Our goal is to have a worker per resource used for training.
    # The priority is GPUs, but can fall back to CPUs if there are no
    # GPUs available.
    use_gpu = kwargs.get("use_gpu", int(ray.cluster_resources().get("GPU", 0)) > 0)
    if use_gpu:
        num_workers = int(ray.cluster_resources().get("GPU", 0))
    else:
        num_workers = _num_nodes()

    strategy = kwargs.pop("strategy", get_default_strategy_name())
    backend = get_dist_strategy(strategy).get_ray_trainer_backend(**kwargs)

    # Remove params used by strategy but not the trainer here
    kwargs.pop("nics", None)

    defaults = dict(
        backend=backend,
        strategy=strategy,
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={
            "CPU": 0 if use_gpu else 1,
            "GPU": 1 if use_gpu else 0,
        },
    )
    return {**defaults, **kwargs}


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


def _local_size() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def train_fn(
    executable_kwargs: Dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: TrainingSetMetadataDict = None,
    features: Dict[str, Dict] = None,
    **kwargs,
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch(local_rank=session.get_local_rank(), local_size=_local_size())
    distributed = get_current_dist_strategy(allow_local=False)()
    try:
        train_shard = RayDatasetShard(
            session.get_dataset_shard("train"),
            features,
            training_set_metadata,
        )

        try:
            val_shard = session.get_dataset_shard("val")
        except KeyError:
            val_shard = None

        if val_shard is not None:
            val_shard = RayDatasetShard(val_shard, features, training_set_metadata)

        try:
            test_shard = session.get_dataset_shard("test")
        except KeyError:
            test_shard = None

        if test_shard is not None:
            test_shard = RayDatasetShard(test_shard, features, training_set_metadata)

        model = ray.get(model_ref)
        device = get_torch_device()
        model = model.to(device)

        trainer = RemoteTrainer(model=model, distributed=distributed, report_tqdm_to_ray=True, **executable_kwargs)
        results = trainer.train(train_shard, val_shard, test_shard, **kwargs)

        if results is not None:
            # only return the model state dict back to the head node.
            trained_model, *args = results
            results = (trained_model.cpu().state_dict(), *args)

        torch.cuda.empty_cache()

        # Passing objects containing Torch tensors as metrics is not supported as it will throw an
        # exception on deserialization, so create a checkpoint and return via session.report() along
        # with the path of the checkpoint
        ckpt = Checkpoint.from_dict({"state_dict": results})
        torch_ckpt = TorchCheckpoint.from_checkpoint(ckpt)

        # The checkpoint is put in the object store and then retrieved by the Trainable actor to be reported to Tune.
        # It is also persisted on disk by the Trainable (and synced to cloud, if configured to do so)
        # The result object returned from trainer.fit() contains the metrics from the last session.report() call.
        # So, make a final call to session.report with the train_results object above.
        session.report(
            metrics={
                "validation_field": trainer.validation_field,
                "validation_metric": trainer.validation_metric,
            },
            checkpoint=torch_ckpt,
        )

    except Exception:
        logger.exception("Exception raised during training by one of the workers")
        raise

    finally:
        distributed.shutdown()


@ray.remote(max_calls=1)
def tune_batch_size_fn(
    dataset: RayDataset = None,
    data_loader_kwargs: Dict[str, Any] = None,
    executable_kwargs: Dict[str, Any] = None,
    model: ECD = None,  # noqa: F821
    ludwig_config: ModelConfigDict = None,
    training_set_metadata: TrainingSetMetadataDict = None,
    features: Dict[str, Dict] = None,
    **kwargs,
) -> int:
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    #
    # As of Ray >= 2.1, to use ray.air.session.get_local_rank(), you need to be inside a train session
    # or a tune session. In Ludwig's current code implementation, batch size tuning doesn't get instantiated
    # inside of a RayTrainer class, so we manually set the local_rank to 0 so that it picks up the right
    # device to tune batch size on.
    initialize_pytorch(local_rank=0, local_size=_local_size())
    distributed = get_current_dist_strategy(allow_local=True)()
    try:
        train_shard = RayDatasetShard(
            dataset.ds,
            features,
            training_set_metadata,
        )

        device = get_torch_device()
        model = model.to(device)

        trainer = RemoteTrainer(model=model, distributed=distributed, **executable_kwargs)
        return trainer.tune_batch_size(ludwig_config, train_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()
        distributed.shutdown()


@DeveloperAPI
class TqdmCallback(ray.tune.callback.Callback):
    """Class for a custom Ray callback that updates tqdm progress bars in the driver process."""

    def __init__(self) -> None:
        """Constructor for TqdmCallback."""
        super().__init__()
        self.progress_bars = {}

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """Called after receiving a result from a trial
        https://docs.ray.io/en/latest/_modules/ray/tune/callback.html#Callback.on_trial_result."""
        progress_bar_opts = result.get("progress_bar")
        if not progress_bar_opts:
            return
        # Skip commands received by non-coordinators
        if not progress_bar_opts["is_coordinator"]:
            return
        _id = progress_bar_opts["id"]
        action = progress_bar_opts.pop("action")
        if action == "create":
            progress_bar_config = progress_bar_opts.get("config")
            self.progress_bars[_id] = tqdm.tqdm(**progress_bar_config)
        elif action == "close":
            self.progress_bars[_id].close()
        elif action == "update":
            update_by = progress_bar_opts.pop("update_by")
            self.progress_bars[_id].update(update_by)


@contextlib.contextmanager
def create_runner(**kwargs):
    trainer_kwargs = get_trainer_kwargs(**kwargs)
    yield RayAirRunner(trainer_kwargs)


class RayAirRunner:
    def __init__(self, trainer_kwargs: Dict[str, Any]) -> None:
        trainer_kwargs = copy.copy(trainer_kwargs)
        self.backend_config = trainer_kwargs.pop("backend", None)
        self.strategy = trainer_kwargs.pop("strategy", get_default_strategy_name())

        if "max_retries" in trainer_kwargs:
            logger.warning("`max_retries` is no longer supported as a trainer argument in Ray backend. Ignoring it.")
            del trainer_kwargs["max_retries"]

        # When training on GPU, you want to pack workers together to limit network latency during
        # allreduce. Conversely, for CPU training you want to spread out the workers to limit
        # CPU and memory contention and avoid too many workers on a single machine.
        strategy = "PACK" if trainer_kwargs.get("use_gpu", False) else "SPREAD"
        # Ray Tune automatically creates a PlacementGroupFactory from the ScalingConfig internally
        self.scaling_config = ScalingConfig(
            placement_strategy=strategy,
            # Override the default of 1 to prevent unnecessary CPU usage.
            trainer_resources={"CPU": 0},
            **trainer_kwargs,
        )

    def _get_dataset_configs(
        self,
        datasets: Dict[str, Any],
        stream_window_size: Dict[str, Union[None, float]],
        data_loader_kwargs: Dict[str, Any],
    ) -> Dict[str, DatasetConfig]:
        """Generates DatasetConfigs for each dataset passed into the trainer."""
        dataset_configs = {}
        for dataset_name, _ in datasets.items():
            if _ray230:
                # DatasetConfig.use_stream_api and DatasetConfig.stream_window_size have been removed as of Ray 2.3.
                # We need to use DatasetConfig.max_object_store_memory_fraction instead -> default to 20% when windowing
                # is enabled unless the end user specifies a different fraction.
                # https://docs.ray.io/en/master/ray-air/check-ingest.html?highlight=max_object_store_memory_fraction#enabling-streaming-ingest # noqa
                dataset_conf = DatasetConfig(
                    split=True,
                    max_object_store_memory_fraction=stream_window_size.get(dataset_name),
                )
            else:
                dataset_conf = DatasetConfig(
                    split=True,
                    use_stream_api=True,
                    stream_window_size=stream_window_size.get(dataset_name),
                )

            if dataset_name == "train":
                # Mark train dataset as always required
                dataset_conf.required = True
                # Check data loader kwargs to see if shuffle should be enabled for the
                # train dataset. global_shuffle is False by default for all other datasets.
                dataset_conf.global_shuffle = data_loader_kwargs.get("shuffle", True)
            dataset_configs[dataset_name] = dataset_conf
        return dataset_configs

    def run(
        self,
        train_loop_per_worker: Callable,
        config: Dict[str, Any],
        dataset: Dict[str, Any],
        data_loader_kwargs: Dict[str, Any],
        stream_window_size: Dict[str, Union[None, float]],
        callbacks: List[Any] = [],
    ) -> Tuple[Dict, TorchCheckpoint]:
        trainer_cls, kwargs = get_dist_strategy(self.strategy).get_trainer_cls(self.backend_config)
        trainer = trainer_cls(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=config,
            datasets=dataset,
            scaling_config=self.scaling_config,
            dataset_config=self._get_dataset_configs(dataset, stream_window_size, data_loader_kwargs),
            run_config=RunConfig(callbacks=callbacks, verbose=0),
            **kwargs,
        )
        return trainer.fit()


@register_ray_trainer(MODEL_ECD, default=True)
class RayTrainerV2(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        trainer_kwargs: Dict[str, Any],
        data_loader_kwargs: Dict[str, Any],
        executable_kwargs: Dict[str, Any],
        **kwargs,
    ):
        self.model = model.cpu()
        self.data_loader_kwargs = data_loader_kwargs or {}
        self.executable_kwargs = executable_kwargs
        self.trainer_kwargs = trainer_kwargs
        self._validation_field = None
        self._validation_metric = None

    @staticmethod
    def get_schema_cls():
        return ECDTrainerConfig

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

        dataset = {"train": training_set.ds}
        stream_window_size = {"train": training_set.window_size_bytes}
        if validation_set is not None:
            dataset["val"] = validation_set.ds
            stream_window_size["val"] = validation_set.window_size_bytes
        if test_set is not None:
            dataset["test"] = test_set.ds
            stream_window_size["test"] = test_set.window_size_bytes

        with create_runner(**self.trainer_kwargs) as runner:
            trainer_results = runner.run(
                lambda config: train_fn(**config),
                config={
                    "executable_kwargs": executable_kwargs,
                    "model_ref": ray.put(self.model),
                    **kwargs,
                },
                callbacks=[TqdmCallback()],
                data_loader_kwargs=self.data_loader_kwargs,
                dataset=dataset,
                stream_window_size=stream_window_size,
            )

        # Set validation field and metric used by trainer
        self._validation_field = trainer_results.metrics["validation_field"]
        self._validation_metric = trainer_results.metrics["validation_metric"]

        # Load model from checkpoint
        ckpt = TorchCheckpoint.from_checkpoint(trainer_results.checkpoint)
        results = ckpt.to_dict()["state_dict"]

        # load state dict back into the model
        state_dict, *args = results
        self.model.load_state_dict(state_dict)
        results = (self.model, *args)

        return results

    def train_online(self, *args, **kwargs):
        # TODO: When this is implemented we also need to update the
        # Tqdm flow to report back the callback
        raise NotImplementedError()

    def tune_batch_size(
        self,
        config: ModelConfigDict,
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

    @property
    def validation_field(self):
        return self._validation_field

    @property
    def validation_metric(self):
        return self._validation_metric

    @property
    def config(self) -> ECDTrainerConfig:
        return self.executable_kwargs["config"]

    @property
    def batch_size(self) -> int:
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self.config.batch_size = value

    @property
    def eval_batch_size(self) -> int:
        return self.config.eval_batch_size if self.config.eval_batch_size is not None else self.config.batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, value: int):
        self.config.eval_batch_size = value

    @property
    def resources_per_worker(self) -> Dict[str, Any]:
        trainer_kwargs = get_trainer_kwargs(**self.trainer_kwargs)
        return trainer_kwargs.get("resources_per_worker", {})

    @property
    def num_cpus(self) -> int:
        return self.resources_per_worker.get("CPU", 1)

    @property
    def num_gpus(self) -> int:
        return self.resources_per_worker.get("GPU", 0)

    def shutdown(self):
        pass


class HorovodRemoteTrainer(RemoteTrainer):
    def __init__(self, **kwargs):
        distributed = get_current_dist_strategy(allow_local=False)()
        super().__init__(distributed=distributed, **kwargs)


def eval_fn(
    predictor_kwargs: Dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: TrainingSetMetadataDict = None,
    features: Dict[str, Dict] = None,
    **kwargs,
):
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch(local_rank=session.get_local_rank(), local_size=_local_size())
    distributed = get_current_dist_strategy(allow_local=False)()
    try:
        eval_shard = RayDatasetShard(
            session.get_dataset_shard("eval"),
            features,
            training_set_metadata,
        )

        model = ray.get(model_ref)
        device = get_torch_device()
        model = model.to(device)

        predictor = RemotePredictor(model=model, distributed=distributed, report_tqdm_to_ray=True, **predictor_kwargs)
        results = predictor.batch_evaluation(eval_shard, **kwargs)

        # The result object returned from trainer.fit() contains the metrics from the last session.report() call.
        # So, make a final call to session.report with the eval_results object above.
        session.report(metrics={"eval_results": results})
    finally:
        torch.cuda.empty_cache()
        distributed.shutdown()


class RayPredictor(BasePredictor):
    def __init__(
        self, model: BaseModel, df_engine: DataFrameEngine, trainer_kwargs, data_loader_kwargs, **predictor_kwargs
    ):
        self.batch_size = predictor_kwargs["batch_size"]
        self.trainer_kwargs = trainer_kwargs
        self.data_loader_kwargs = data_loader_kwargs
        self.predictor_kwargs = predictor_kwargs
        self.actor_handles = []
        self.model = model.cpu()
        self.df_engine = df_engine

    def get_trainer_kwargs(self) -> Dict[str, Any]:
        return get_trainer_kwargs(**self.trainer_kwargs)

    def get_resources_per_worker(self) -> Tuple[int, int]:
        trainer_kwargs = self.get_trainer_kwargs()
        resources_per_worker = trainer_kwargs.get("resources_per_worker", {})
        num_gpus = resources_per_worker.get("GPU", 0)
        num_cpus = resources_per_worker.get("CPU", (1 if num_gpus == 0 else 0))
        return num_cpus, num_gpus

    def batch_predict(self, dataset: RayDataset, *args, collect_logits: bool = False, **kwargs):
        self._check_dataset(dataset)

        predictor_kwargs = self.predictor_kwargs
        output_columns = get_output_columns(self.model.output_features, include_logits=collect_logits)
        batch_predictor = self.get_batch_infer_model(
            self.model,
            predictor_kwargs,
            output_columns,
            dataset.features,
            dataset.training_set_metadata,
            *args,
            collect_logits=collect_logits,
            **kwargs,
        )

        num_cpus, num_gpus = self.get_resources_per_worker()

        with tensor_extension_casting(False):
            predictions = dataset.ds.map_batches(
                batch_predictor,
                batch_size=self.batch_size,
                compute="actors",
                batch_format="pandas",
                num_cpus=num_cpus,
                num_gpus=num_gpus,
            )
            predictions = self.df_engine.from_ray_dataset(predictions)

        for of_feature in self.model.output_features.values():
            predictions = of_feature.unflatten(predictions)

        return predictions

    def predict_single(self, batch):
        raise NotImplementedError("predict_single can only be called on a local predictor")

    def batch_evaluation(
        self,
        dataset: RayDataset,
        collect_predictions: bool = False,
        collect_logits=False,
        **kwargs,
    ):
        # We need to be in a Horovod context to collect the aggregated metrics, since it relies on collective
        # communication ops. However, Horovod is not suitable for transforming one big dataset to another. For that
        # we will use Ray Datasets. Therefore, we break this up into two separate steps, and two passes over the
        # dataset. In the future, we can explore ways to combine these into a single step to reduce IO.
        with create_runner(**self.trainer_kwargs) as runner:
            # Collect eval metrics by distributing work across nodes / gpus with Horovod
            datasets = {"eval": dataset.ds}
            stream_window_size = {"eval": dataset.window_size_bytes}
            predictor_kwargs = {**self.predictor_kwargs, "collect_predictions": False}
            eval_results = runner.run(
                lambda config: eval_fn(**config),
                config={
                    "predictor_kwargs": predictor_kwargs,
                    "model_ref": ray.put(self.model),
                    "training_set_metadata": dataset.training_set_metadata,
                    "features": dataset.features,
                    **kwargs,
                },
                dataset=datasets,
                data_loader_kwargs=self.data_loader_kwargs,
                stream_window_size=stream_window_size,
            )

        eval_stats = eval_results.metrics["eval_results"][0]

        predictions = None
        if collect_predictions:
            # Collect eval predictions by using Ray Datasets to transform partitions of the data in parallel
            predictions = self.batch_predict(dataset, collect_logits=collect_logits)

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
        training_set_metadata: TrainingSetMetadataDict,
        *args,
        **kwargs,
    ):
        model_ref = ray.put(model)

        class BatchInferModel:
            def __init__(self):
                model = ray.get(model_ref)
                device = get_torch_device()
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
                predictions = self.predict(batch=dataset).set_index(df.index)
                for output_feature in self.model.output_features.values():
                    predictions = output_feature.flatten(predictions)
                ordered_predictions = predictions[self.output_columns]
                return ordered_predictions

            def _prepare_batch(self, batch: pd.DataFrame) -> Dict[str, np.ndarray]:
                res = {}
                for c in self.features.keys():
                    if self.features[c][TYPE] not in _SCALAR_TYPES:
                        # Ensure columns stacked instead of turned into np.array([np.array, ...], dtype=object) objects
                        res[c] = np.stack(batch[c].values)
                    else:
                        res[c] = batch[c].to_numpy()

                for c in self.features.keys():
                    reshape = self.reshape_map.get(c)
                    if reshape is not None:
                        res[c] = res[c].reshape((-1, *reshape))

                return res

        return BatchInferModel


class RayBackend(RemoteTrainingMixin, Backend):
    BACKEND_TYPE = "ray"

    def __init__(self, processor=None, trainer=None, loader=None, preprocessor_kwargs=None, **kwargs):
        super().__init__(dataset_manager=RayDatasetManager(self), **kwargs)
        self._preprocessor_kwargs = preprocessor_kwargs or {}
        self._df_engine = _get_df_engine(processor)
        self._horovod_kwargs = trainer or {}
        self._pytorch_kwargs = {}
        self._data_loader_kwargs = loader or {}
        self._preprocessor_pg = None

    def initialize(self):
        initialize_ray()

        dask.config.set(scheduler=ray_dask_get)
        # Disable placement groups on dask
        dask.config.set(annotations={"ray_remote_args": {"placement_group": None}})

    def generate_bundles(self, num_cpu):
        # Ray requires that each bundle be scheduleable on a single node.
        # So a bundle of 320 cpus would never get scheduled. For now a simple heuristic
        # to be used is to just request 1 cpu at a time.
        return [{"CPU": 1} for _ in range(int(num_cpu))]

    @contextlib.contextmanager
    def provision_preprocessing_workers(self):
        num_cpu = self._preprocessor_kwargs.get("num_cpu")
        if not num_cpu:
            logger.info(
                "Backend config has num_cpu not set." " provision_preprocessing_workers() is a no-op in this case."
            )
            yield
        else:
            bundles = self.generate_bundles(num_cpu)
            logger.info("Requesting bundles of %s for preprocessing", bundles)
            self._preprocessor_pg = placement_group(bundles)
            ready = self._preprocessor_pg.wait(FIFTEEN_MINS_IN_S)

            if not ready:
                remove_placement_group(self._preprocessor_pg)
                raise TimeoutError(
                    "Ray timed out in provisioning the placement group for preprocessing."
                    f" {num_cpu} CPUs were requested but were unable to be provisioned."
                )

            logger.info("%s CPUs were requested and successfully provisioned", num_cpu)
            try:
                with dask.config.set(annotations={"ray_remote_args": {"placement_group": self._preprocessor_pg}}):
                    yield
            finally:
                self._release_preprocessing_workers()

    def _release_preprocessing_workers(self):
        if self._preprocessor_pg is not None:
            remove_placement_group(self._preprocessor_pg)
        self._preprocessor_pg = None

    def initialize_pytorch(self, **kwargs):
        # Make sure we don't claim any GPU resources on the head node
        initialize_pytorch(gpus=-1)
        self._pytorch_kwargs = kwargs

    def create_trainer(self, model: BaseModel, **kwargs) -> "BaseTrainer":  # noqa: F821
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}

        trainer_cls = get_from_registry(model.type(), ray_trainers_registry)

        all_kwargs = {
            "model": model,
            "trainer_kwargs": self._horovod_kwargs,
            "data_loader_kwargs": self._data_loader_kwargs,
            "executable_kwargs": executable_kwargs,
        }
        all_kwargs.update(kwargs)

        return trainer_cls(**all_kwargs)

    def create_predictor(self, model: BaseModel, **kwargs):
        executable_kwargs = {**kwargs, **self._pytorch_kwargs}
        return RayPredictor(
            model,
            self.df_engine,
            self._horovod_kwargs,
            self._data_loader_kwargs,
            **executable_kwargs,
        )

    @property
    def distributed_kwargs(self):
        return self._horovod_kwargs

    @distributed_kwargs.setter
    def distributed_kwargs(self, value):
        self._horovod_kwargs = value

    @property
    def df_engine(self):
        return self._df_engine

    @property
    def supports_multiprocessing(self):
        return False

    def read_binary_files(
        self, column: Series, map_fn: Optional[Callable] = None, file_size: Optional[int] = None
    ) -> Series:
        column = column.fillna(np.nan).replace([np.nan], [None])  # normalize NaNs to None

        # Assume that the list of filenames is small enough to fit in memory. Should be true unless there
        # are literally billions of filenames.
        # TODO(travis): determine if there is a performance penalty to passing in individual files instead of
        #  a directory. If so, we can do some preprocessing to determine if it makes sense to read the full directory
        #  then filter out files as a postprocessing step (depending on the ratio of included to excluded files in
        #  the directory). Based on a preliminary look at how Ray handles directory expansion to files, it looks like
        #  there should not be any difference between providing a directory versus a list of files.
        pd_column = self.df_engine.compute(column)
        fnames = pd_column.values.tolist()
        idxs = pd_column.index.tolist()

        # Sample a filename to extract the filesystem info
        sample_fname = fnames[0]
        if isinstance(sample_fname, str):
            fs, _ = get_fs_and_path(sample_fname)

            read_datasource_fn_kwargs = {
                "path_and_idxs": list(zip(fnames, idxs)),
                "filesystem": PyFileSystem(FSSpecHandler(fs)),
            }
            if self.df_engine.partitioned and file_size is not None:
                # Heuristic to determine parallelism: if the average file size is known (in bytes), then we can
                # extrapolate to determine the total file size. We aim to have ~50MB partitions (5e7 bytes), so we
                # set parallelism to be the total size / 50MB.
                total_size = file_size * len(fnames)
                parallelism = int(total_size / 5e7)
                # Only set parallelism if it matches or exceeds the Ray default kwarg for parallelism
                read_datasource_fn_kwargs["parallelism"] = max(RAY_DEFAULT_PARALLELISM, parallelism)

            # The resulting column is named "value", which is a dict with two keys: "idx" and "data".
            ds = ray.data.read_datasource(BinaryIgnoreNoneTypeDatasource(), **read_datasource_fn_kwargs)
            df = self.df_engine.from_ray_dataset(ds)
            df["idx"] = self.df_engine.map_objects(df["value"], lambda row: int(row["idx"]))
            df["value"] = self.df_engine.map_objects(df["value"], lambda row: row["data"])
            df = df.rename(columns={"value": column.name})
        else:
            # Assume the path has already been read in, so just convert directly to a dataset
            # Name the column "value" to match the behavior of the above
            df = column.to_frame(name=column.name)
            df["idx"] = df.index

        if map_fn is not None:
            df[column.name] = self.df_engine.map_objects(df[column.name], map_fn)

        if "idx" in df.columns:
            df = df.set_index("idx", drop=True)
            df = self.df_engine.map_partitions(
                df, lambda pd_df: set_index_name(pd_df, column.index.name), meta={column.name: "object"}
            )
        return df[column.name]

    @property
    def num_nodes(self) -> int:
        if not ray.is_initialized():
            return 1
        return len(ray.nodes())

    def get_available_resources(self) -> Resources:
        resources = ray.cluster_resources()
        return Resources(cpus=resources.get("CPU", 0), gpus=resources.get("GPU", 0))

    def max_concurrent_trials(self, hyperopt_config: HyperoptConfigDict) -> Union[int, None]:
        cpus_per_trial = hyperopt_config[EXECUTOR].get(CPU_RESOURCES_PER_TRIAL, 1)
        num_cpus_available = self.get_available_resources().cpus

        # No actors will compete for ray datasets tasks dataset tasks are cpu bound
        if cpus_per_trial == 0:
            return None

        if num_cpus_available < 2:
            logger.warning(
                "At least 2 CPUs are required for hyperopt when using a RayBackend, but only found "
                f"{num_cpus_available}. If you are not using an auto-scaling Ray cluster, your hyperopt "
                "trials may hang."
            )

        # Ray requires at least 1 free CPU to ensure trials don't stall
        max_possible_trials = int(num_cpus_available // cpus_per_trial) - 1

        # Users may be using an autoscaling cluster, so return None
        if max_possible_trials < 1:
            logger.warning(
                f"Hyperopt trials will request {cpus_per_trial} CPUs in addition to CPUs needed for Ray Datasets, "
                f" but only {num_cpus_available} CPUs are currently available. If you are not using an auto-scaling "
                " Ray cluster, your hyperopt trials may hang."
            )
            return None

        return max_possible_trials

    def tune_batch_size(self, evaluator_cls: Type[BatchSizeEvaluator], dataset_len: int) -> int:
        return ray.get(
            _tune_batch_size_fn.options(**self._get_transform_kwargs()).remote(
                evaluator_cls,
                dataset_len,
            )
        )

    def batch_transform(self, df: DataFrame, batch_size: int, transform_fn: Callable) -> DataFrame:
        ds = self.df_engine.to_ray_dataset(df)
        ds = ds.map_batches(
            transform_fn, batch_size=batch_size, compute="actors", batch_format="pandas", **self._get_transform_kwargs()
        )
        return self.df_engine.from_ray_dataset(ds)

    def _get_transform_kwargs(self) -> Dict[str, Any]:
        trainer_kwargs = get_trainer_kwargs(**self._horovod_kwargs)
        resources_per_worker = trainer_kwargs.get("resources_per_worker", {})
        num_gpus = resources_per_worker.get("GPU", 0)
        num_cpus = resources_per_worker.get("CPU", (1 if num_gpus == 0 else 0))
        return dict(num_cpus=num_cpus, num_gpus=num_gpus)


@ray.remote(max_calls=1)
def _tune_batch_size_fn(evaluator_cls: Type[BatchSizeEvaluator], dataset_len: int) -> int:
    evaluator = evaluator_cls()
    return evaluator.select_best_batch_size(dataset_len)


def initialize_ray():
    if not ray.is_initialized():
        try:
            ray.init("auto", ignore_reinit_error=True)
        except ConnectionError:
            init_ray_local()


def init_ray_local():
    logger.info("Initializing new Ray cluster...")
    ray.init(ignore_reinit_error=True)
