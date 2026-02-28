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
import os
import tempfile
from collections.abc import Callable
from functools import partial
from typing import Any, TYPE_CHECKING

import dask
import numpy as np
import pandas as pd
import ray
import ray.train as rt
import torch
import tqdm
from fsspec.config import conf
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray import ObjectRef
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.constants import TRAIN_ENABLE_WORKER_SPREAD_ENV
from ray.train.torch import TorchConfig, TorchTrainer
from ray.util.dask import ray_dask_get
from ray.util.placement_group import placement_group, remove_placement_group

if TYPE_CHECKING:
    from ludwig.api import LudwigModel

from ludwig.backend.base import Backend, RemoteTrainingMixin
from ludwig.backend.datasource import read_binary_files_with_index
from ludwig.constants import MODEL_ECD, MODEL_LLM, NAME, PREPROCESSING, PROC_COLUMN, TYPE
from ludwig.data.dataframe.base import DataFrameEngine

try:
    from ludwig.data.dataset.ray import (
        _SCALAR_TYPES,
        cast_as_tensor_dtype,
        RayDataset,
        RayDatasetManager,
        RayDatasetShard,
    )
except (ImportError, AttributeError):
    _SCALAR_TYPES = cast_as_tensor_dtype = RayDataset = RayDatasetManager = RayDatasetShard = None
from ludwig.models.base import BaseModel
from ludwig.models.ecd import ECD
from ludwig.models.predictor import BasePredictor, get_output_columns, get_predictor_cls
from ludwig.schema.trainer import ECDTrainerConfig
from ludwig.trainers.registry import get_ray_trainers_registry, register_ray_trainer
from ludwig.trainers.trainer import BaseTrainer, RemoteTrainer
from ludwig.utils.data_utils import use_credentials
from ludwig.utils.fs_utils import get_fs_and_path
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.system_utils import Resources
from ludwig.utils.torch_utils import get_torch_device, initialize_pytorch
from ludwig.utils.types import Series

logger = logging.getLogger(__name__)

FIFTEEN_MINS_IN_S = 15 * 60


def _num_nodes() -> int:
    node_resources = [node["Resources"] for node in ray.nodes()]
    return len(node_resources)


def get_trainer_kwargs(**kwargs) -> dict[str, Any]:
    kwargs = copy.deepcopy(kwargs)

    # Our goal is to have a worker per resource used for training.
    # The priority is GPUs, but can fall back to CPUs if there are no
    # GPUs available.
    use_gpu = kwargs.get("use_gpu", int(ray.cluster_resources().get("GPU", 0)) > 0)
    if use_gpu:
        num_workers = int(ray.cluster_resources().get("GPU", 0))
    else:
        num_workers = _num_nodes()

    # Remove nics if present (legacy option)
    kwargs.pop("nics", None)

    defaults = dict(
        backend=TorchConfig(),
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


def _make_picklable(obj):
    """Recursively convert defaultdicts (which contain unpicklable lambdas) to regular dicts."""
    from collections import defaultdict

    if isinstance(obj, defaultdict):
        return {k: _make_picklable(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _make_picklable(v) for k, v in obj.items()}
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # NamedTuple: reconstruct with the same field names
        return type(obj)(**{f: _make_picklable(getattr(obj, f)) for f in obj._fields})
    elif isinstance(obj, list):
        return [_make_picklable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_picklable(item) for item in obj)
    return obj


def train_fn(
    executable_kwargs: dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: dict[str, Any] = None,
    features: dict[str, dict] = None,
    **kwargs,
):
    """Ray Train worker function for distributed training.

    Runs inside each Ray worker process. Loads the model from an object ref, wraps dataset shards, trains, and saves
    results to a Ray checkpoint so the driver can retrieve them (Ray Train 2.x requires a checkpoint for metrics).
    """
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch()

    # Initialize a local distributed strategy so metric modules can sync.
    from ludwig.distributed import init_dist_strategy

    init_dist_strategy("local")

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
    # Use Ray Train's device assignment which respects use_gpu setting,
    # rather than get_torch_device() which always picks CUDA if available.
    from ray.train.torch import get_device as ray_get_device

    device = ray_get_device()
    model = model.to(device)

    trainer = RemoteTrainer(model=model, report_tqdm_to_ray=True, **executable_kwargs)
    results = trainer.train(train_shard, val_shard, test_shard, **kwargs)

    if results is not None:
        # only return the model state dict back to the head node.
        trained_model, *args = results
        results = (trained_model.cpu().state_dict(), *args)

    torch.cuda.empty_cache()

    # Save results to a checkpoint so the driver can retrieve them.
    # In Ray Train 2.x, result.metrics is only populated when a checkpoint is provided.
    train_results = results, trainer.validation_field, trainer.validation_metric
    # Convert defaultdicts to regular dicts so they can be pickled by torch.save.
    train_results = _make_picklable(train_results)
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(train_results, os.path.join(tmpdir, "train_results.pt"))
        rt.report(metrics={}, checkpoint=Checkpoint.from_directory(tmpdir))


@ray.remote
def tune_batch_size_fn(
    dataset: RayDataset = None,
    data_loader_kwargs: dict[str, Any] = None,
    executable_kwargs: dict[str, Any] = None,
    model: ECD = None,  # noqa: F821
    ludwig_config: dict[str, Any] = None,
    training_set_metadata: dict[str, Any] = None,
    features: dict[str, dict] = None,
    **kwargs,
) -> int:
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch()

    try:
        ds = dataset.to_ray_dataset(shuffle=False)
        train_shard = RayDatasetShard(
            ds,
            features,
            training_set_metadata,
        )

        device = get_torch_device()
        model = model.to(device)

        trainer = RemoteTrainer(model=model, **executable_kwargs)
        return trainer.tune_batch_size(ludwig_config, train_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()


@ray.remote
def tune_learning_rate_fn(
    dataset: RayDataset,
    config: dict[str, Any],
    data_loader_kwargs: dict[str, Any] = None,
    executable_kwargs: dict[str, Any] = None,
    model: ECD = None,  # noqa: F821
    training_set_metadata: dict[str, Any] = None,
    features: dict[str, dict] = None,
    **kwargs,
) -> float:
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch()

    try:
        ds = dataset.to_ray_dataset(shuffle=False)
        train_shard = RayDatasetShard(
            ds,
            features,
            training_set_metadata,
        )

        device = get_torch_device()
        model = model.to(device)

        trainer = RemoteTrainer(model=model, **executable_kwargs)
        return trainer.tune_learning_rate(config, train_shard, **kwargs)
    finally:
        torch.cuda.empty_cache()


class TqdmCallback(rt.UserCallback):
    """Class for a custom ray callback that updates tqdm progress bars in the driver process."""

    def __init__(self) -> None:
        """Constructor for TqdmCallback."""
        super().__init__()
        self.progess_bars = {}

    def after_report(self, run_context, metrics: list[dict], checkpoint=None) -> None:
        """Called every time ray.train.report is called from subprocesses.

        In Ray 2.x, metrics is a list of metric dicts (one per worker). We look for progress_bar data from the
        coordinator worker.
        """
        for result in metrics:
            progress_bar_opts = result.get("progress_bar")
            if not progress_bar_opts:
                continue
            # Skip commands received by non-coordinators
            if not progress_bar_opts["is_coordinator"]:
                continue
            _id = progress_bar_opts["id"]
            action = progress_bar_opts.get("action")
            if action == "create":
                progress_bar_config = progress_bar_opts.get("config")
                self.progess_bars[_id] = tqdm.tqdm(**progress_bar_config)
            elif action == "close":
                if _id in self.progess_bars:
                    self.progess_bars[_id].close()
            elif action == "update":
                update_by = progress_bar_opts.get("update_by", 1)
                if _id in self.progess_bars:
                    self.progess_bars[_id].update(update_by)


@contextlib.contextmanager
def spread_env(use_gpu: bool = False, num_workers: int = 1, **kwargs):
    if TRAIN_ENABLE_WORKER_SPREAD_ENV in os.environ:
        # User set this explicitly, so honor their selection
        yield
        return

    try:
        if not use_gpu and num_workers > 1:
            # When doing CPU-only training, default to a SPREAD policy to avoid
            # packing too many workers on a single machine
            os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV] = "1"
        yield
    finally:
        if TRAIN_ENABLE_WORKER_SPREAD_ENV in os.environ:
            del os.environ[TRAIN_ENABLE_WORKER_SPREAD_ENV]


def _build_scaling_config(trainer_kwargs: dict[str, Any]) -> ScalingConfig:
    """Convert legacy trainer kwargs to a Ray ScalingConfig."""
    return ScalingConfig(
        num_workers=trainer_kwargs.get("num_workers", 1),
        use_gpu=trainer_kwargs.get("use_gpu", False),
        resources_per_worker=trainer_kwargs.get("resources_per_worker"),
    )


def run_train_remote(train_loop, trainer_kwargs: dict[str, Any], callbacks=None, datasets=None, train_loop_config=None):
    """Run a distributed training function using Ray TorchTrainer."""
    resolved_kwargs = get_trainer_kwargs(**trainer_kwargs)

    scaling_config = _build_scaling_config(resolved_kwargs)
    torch_config = resolved_kwargs.get("backend", TorchConfig())

    run_config_kwargs = {}
    if callbacks:
        run_config_kwargs["callbacks"] = callbacks

    with spread_env(**resolved_kwargs):
        torch_trainer = TorchTrainer(
            train_loop_per_worker=train_loop,
            train_loop_config=train_loop_config,
            torch_config=torch_config,
            scaling_config=scaling_config,
            run_config=RunConfig(**run_config_kwargs),
            datasets=datasets,
        )
        result = torch_trainer.fit()
    return result


@register_ray_trainer(MODEL_ECD, default=True)
class RayTrainerV2(BaseTrainer):
    def __init__(
        self,
        model: BaseModel,
        trainer_kwargs: dict[str, Any],
        data_loader_kwargs: dict[str, Any],
        executable_kwargs: dict[str, Any],
        **kwargs,
    ):
        self.model = model.cpu()
        self.data_loader_kwargs = data_loader_kwargs
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
        validation_set: RayDataset | None = None,
        test_set: RayDataset | None = None,
        **kwargs,
    ):
        executable_kwargs = self.executable_kwargs

        kwargs = {
            "training_set_metadata": training_set.training_set_metadata,
            "features": training_set.features,
            **kwargs,
        }

        dataset = {"train": training_set.to_ray_dataset(shuffle=True)}
        if validation_set is not None:
            dataset["val"] = validation_set.to_ray_dataset(shuffle=False)
        if test_set is not None:
            dataset["test"] = test_set.to_ray_dataset(shuffle=False)

        train_loop_config = {"executable_kwargs": executable_kwargs, "model_ref": ray.put(self.model), **kwargs}

        def _train_loop(config):
            train_fn(**config)

        result = run_train_remote(
            _train_loop,
            trainer_kwargs=self.trainer_kwargs,
            callbacks=[TqdmCallback()],
            datasets=dataset,
            train_loop_config=train_loop_config,
        )

        # Load training results from the checkpoint saved by train_fn
        with result.checkpoint.as_directory() as tmpdir:
            train_results = torch.load(os.path.join(tmpdir, "train_results.pt"), weights_only=False)
        results, self._validation_field, self._validation_metric = train_results

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
        config: dict[str, Any],
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
    def resources_per_worker(self) -> dict[str, Any]:
        trainer_kwargs = get_trainer_kwargs(**self.trainer_kwargs)
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


def eval_fn(
    predictor_kwargs: dict[str, Any] = None,
    model_ref: ObjectRef = None,  # noqa: F821
    training_set_metadata: dict[str, Any] = None,
    features: dict[str, dict] = None,
    **kwargs,
):
    """Ray Train worker function for distributed evaluation.

    Runs inside each Ray worker process. Loads the model from an object ref, wraps the eval dataset shard, runs
    prediction and evaluation, and saves results to a Ray checkpoint for driver retrieval.
    """
    # Pin GPU before loading the model to prevent memory leaking onto other devices
    initialize_pytorch()

    # Initialize a local distributed strategy so metric modules can sync.
    from ludwig.distributed import init_dist_strategy

    init_dist_strategy("local")

    try:
        eval_shard = RayDatasetShard(
            rt.get_dataset_shard("eval"),
            features,
            training_set_metadata,
        )

        model = ray.get(model_ref)
        # Use Ray Train's device assignment which respects use_gpu setting
        from ray.train.torch import get_device as ray_get_device

        device = ray_get_device()
        model = model.to(device)

        predictor_cls = get_predictor_cls(model.type())
        predictor = predictor_cls(dist_model=model, model=model, report_tqdm_to_ray=True, **predictor_kwargs)
        eval_results = predictor.batch_evaluation(eval_shard, **kwargs)

        # Save results to a checkpoint so the driver can retrieve them.
        # In Ray Train 2.x, result.metrics is only populated when a checkpoint is provided.
        eval_results = _make_picklable(eval_results)
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(eval_results, os.path.join(tmpdir, "eval_results.pt"))
            rt.report(metrics={}, checkpoint=Checkpoint.from_directory(tmpdir))
    finally:
        torch.cuda.empty_cache()


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

    def get_trainer_kwargs(self) -> dict[str, Any]:
        return get_trainer_kwargs(**self.trainer_kwargs)

    def get_resources_per_worker(self) -> tuple[int, int]:
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

        columns = [f.proc_column for f in self.model.input_features.values()]

        def to_tensors(df: pd.DataFrame) -> pd.DataFrame:
            for c in columns:
                df[c] = cast_as_tensor_dtype(df[c])
            return df

        num_cpus, num_gpus = self.get_resources_per_worker()

        predictions = dataset.ds.map_batches(to_tensors, batch_format="pandas").map_batches(
            batch_predictor,
            batch_size=self.batch_size,
            compute=ray.data.ActorPoolStrategy(),
            batch_format="pandas",
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

        predictions = self.df_engine.from_ray_dataset(predictions)

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
        # We need to be in a distributed context to collect the aggregated metrics, since it relies on collective
        # communication ops. However, distributed training is not suitable for transforming one big dataset to another.
        # For that we will use Ray Datasets. Therefore, we break this up into two separate steps, and two passes over
        # the dataset. In the future, we can explore ways to combine these into a single step to reduce IO.
        # Collect eval metrics by distributing work across nodes / gpus
        datasets = {"eval": dataset.to_ray_dataset(shuffle=False)}
        predictor_kwargs = {
            **self.predictor_kwargs,
            "collect_predictions": False,
        }
        eval_loop_config = {
            "predictor_kwargs": predictor_kwargs,
            "model_ref": ray.put(self.model),
            "training_set_metadata": dataset.training_set_metadata,
            "features": dataset.features,
            **kwargs,
        }

        def _eval_loop(config):
            eval_fn(**config)

        result = run_train_remote(
            _eval_loop,
            trainer_kwargs=self.trainer_kwargs,
            datasets=datasets,
            train_loop_config=eval_loop_config,
        )

        # Load eval results from the checkpoint saved by eval_fn
        with result.checkpoint.as_directory() as tmpdir:
            eval_stats, _ = torch.load(os.path.join(tmpdir, "eval_results.pt"), weights_only=False)

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
        predictor_kwargs: dict[str, Any],
        output_columns: list[str],
        features: dict[str, dict],
        training_set_metadata: dict[str, Any],
        *args,
        **kwargs,
    ):
        model_ref = ray.put(model)
        _, num_gpus = self.get_resources_per_worker()

        class BatchInferModel:
            def __init__(self):
                model = ray.get(model_ref)
                # Respect the GPU setting from resources_per_worker.
                # When num_gpus=0, force CPU even if CUDA is available on the machine,
                # to avoid device mismatches between model outputs and targets.
                if num_gpus > 0:
                    device = get_torch_device()
                else:
                    device = "cpu"
                self.model = model.to(device)

                self.output_columns = output_columns
                self.features = features
                self.training_set_metadata = training_set_metadata
                self.reshape_map = {
                    f[PROC_COLUMN]: training_set_metadata[f[NAME]].get("reshape") for f in features.values()
                }
                predictor_cls = get_predictor_cls(self.model.type())
                predictor = predictor_cls(dist_model=self.model, model=self.model, **predictor_kwargs)
                self.predict = partial(predictor.predict_single, *args, **kwargs)

            def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
                dataset = self._prepare_batch(df)
                predictions = self.predict(batch=dataset).set_index(df.index)
                ordered_predictions = predictions[self.output_columns]
                return ordered_predictions

            def _prepare_batch(self, batch: pd.DataFrame) -> dict[str, np.ndarray]:
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
        self._distributed_kwargs = trainer or {}
        self._pytorch_kwargs = {}
        self._data_loader_kwargs = loader or {}
        self._preprocessor_pg = None

    def initialize(self):
        initialize_ray()

        dask.config.set(scheduler=ray_dask_get)
        # Disable placement groups on dask
        dask.config.set(annotations={"ray_remote_args": {"placement_group": None}})
        # Prevent Dask from converting object-dtype columns to PyArrow strings,
        # which corrupts binary data, numpy arrays, and complex Python objects.
        dask.config.set({"dataframe.convert-string": False})

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
        if model.type() == MODEL_LLM:
            from ludwig.trainers.registry import get_llm_ray_trainers_registry

            trainer_config = kwargs.get("config")
            trainer_type = trainer_config.type if trainer_config else None
            trainer_cls = get_from_registry(trainer_type, get_llm_ray_trainers_registry())
        else:
            trainer_cls = get_from_registry(model.type(), get_ray_trainers_registry())

        # Deep copy to workaround https://github.com/ray-project/ray/issues/24139
        all_kwargs = {
            "model": model,
            "trainer_kwargs": copy.deepcopy(self._distributed_kwargs),
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
            copy.deepcopy(self._distributed_kwargs),
            self._data_loader_kwargs,
            **executable_kwargs,
        )

    def set_distributed_kwargs(self, **kwargs):
        self._distributed_kwargs = kwargs

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

    def read_binary_files(self, column: Series, map_fn: Callable | None = None, file_size: int | None = None) -> Series:
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
            filesystem = PyFileSystem(FSSpecHandler(fs))

            paths_and_idxs = list(zip(fnames, idxs))
            ds = read_binary_files_with_index(paths_and_idxs, filesystem=filesystem)
            # Rename "data" column to "value" for downstream compatibility
            ds = ds.rename_columns({"data": "value"})
        else:
            # Assume the path has already been read in, so just convert directly to a dataset
            # Name the column "value" to match the behavior of the above
            column_df = column.to_frame(name="value")
            column_df["idx"] = column_df.index
            ds = self.df_engine.to_ray_dataset(column_df)

        # Collect the Ray Dataset to pandas to avoid Arrow's string coercion
        # for binary/object columns (to_dask() converts bytes to string[pyarrow],
        # corrupting binary data and complex Python objects).
        pdf = ds.to_pandas()

        if map_fn is not None:
            with use_credentials(conf):
                pdf["value"] = pdf["value"].map(map_fn)

        pdf = pdf.rename(columns={"value": column.name})
        if "idx" in pdf.columns:
            pdf = pdf.set_index("idx", drop=True)
            pdf.index.name = column.index.name

        # Convert to Dask for downstream compatibility.
        # Note: dataframe.convert-string is disabled globally in RayBackend.initialize()
        # to prevent object-dtype columns from being coerced to PyArrow strings.
        df = self.df_engine.from_pandas(pdf)
        return df[column.name]

    @property
    def num_nodes(self) -> int:
        if not ray.is_initialized():
            return 1
        return len(ray.nodes())

    @property
    def num_training_workers(self) -> int:
        return self._distributed_kwargs.get("num_workers", 1)

    def max_concurrent_trials(self, hyperopt_config) -> int | None:
        # Limit concurrency based on available resources to avoid deadlocks between
        # Ray Tune trials and the Ray Datasets used internally for distributed training.
        resources = self.get_available_resources()
        num_cpus_per_trial = self._distributed_kwargs.get("resources_per_worker", {}).get("CPU", 1)
        num_workers = self._distributed_kwargs.get("num_workers", 1)
        cpus_per_trial = num_cpus_per_trial * num_workers
        if cpus_per_trial > 0 and resources.cpus > 0:
            return max(1, int(resources.cpus // cpus_per_trial))
        return None

    def tune_batch_size(self, evaluator_cls, dataset_len: int) -> int:
        evaluator = evaluator_cls()
        return evaluator.select_best_batch_size(dataset_len)

    def batch_transform(self, df, batch_size: int, transform_fn, name: str | None = None):
        name = name or "Batch Transform"
        from ludwig.utils.dataframe_utils import from_batches, to_batches

        # Compute Dask DataFrame to pandas before batching, as Dask-expr
        # doesn't support row slicing via integer indexing (df[i:j]).
        df = self.df_engine.compute(df)
        batches = to_batches(df, batch_size)
        transform = transform_fn()
        out_batches = [transform(batch.reset_index(drop=True)) for batch in batches]
        out_df = from_batches(out_batches).reset_index(drop=True)
        return out_df

    def get_available_resources(self) -> Resources:
        resources = ray.cluster_resources()
        return Resources(cpus=resources.get("CPU", 0), gpus=resources.get("GPU", 0))


def initialize_ray():
    if not ray.is_initialized():
        try:
            ray.init("auto", ignore_reinit_error=True)
        except ConnectionError:
            init_ray_local()


def init_ray_local():
    logger.info("Initializing new Ray cluster...")
    ray.init(ignore_reinit_error=True)
