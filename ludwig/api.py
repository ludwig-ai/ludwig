# !/usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
"""
File name: LudwigModel.py
Author: Piero Molino
Date created: 5/21/2019
Python Version: 3+
"""

import copy
import dataclasses
import logging
import os
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pprint import pformat
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from ludwig.api_annotations import PublicAPI
from ludwig.backend import Backend, initialize_backend, provision_preprocessing_workers
from ludwig.callbacks import Callback
from ludwig.constants import (
    AUTO,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    FALLBACK_BATCH_SIZE,
    FULL,
    HYPEROPT,
    HYPEROPT_WARNING,
    MIN_DATASET_SPLIT_ROWS,
    MODEL_ECD,
    MODEL_LLM,
    TEST,
    TIMESERIES,
    TRAINING,
    VALIDATION,
)
from ludwig.data.cache.types import CacheableDataset
from ludwig.data.dataset.base import Dataset
from ludwig.data.postprocessing import convert_predictions, postprocess
from ludwig.data.preprocessing import load_metadata, preprocess_for_prediction, preprocess_for_training
from ludwig.datasets import load_dataset_uris
from ludwig.features.feature_registries import update_config_with_metadata, update_config_with_model
from ludwig.features.timeseries_feature import incremental_time_delay_embedding
from ludwig.globals import (
    LUDWIG_VERSION,
    MODEL_FILE_NAME,
    MODEL_HYPERPARAMETERS_FILE_NAME,
    model_weights_exist,
    MODEL_WEIGHTS_FILE_NAME,
    set_disable_progressbar,
    TRAIN_SET_METADATA_FILE_NAME,
    TRAINING_CHECKPOINTS_DIR_PATH,
)
from ludwig.models.base import BaseModel
from ludwig.models.calibrator import Calibrator
from ludwig.models.predictor import (
    calculate_overall_stats,
    print_evaluation_stats,
    save_evaluation_stats,
    save_prediction_outputs,
)
from ludwig.models.registry import model_type_registry
from ludwig.schema.model_config import ModelConfig
from ludwig.types import ModelConfigDict, TrainingSetMetadataDict
from ludwig.upload import get_upload_registry
from ludwig.utils import metric_utils
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.config_utils import get_preprocessing_params
from ludwig.utils.data_utils import (
    figure_data_format,
    generate_kfold_splits,
    load_dataset,
    load_json,
    load_yaml,
    save_json,
)
from ludwig.utils.dataset_utils import generate_dataset_statistics
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import makedirs, path_exists, upload_output_directory
from ludwig.utils.heuristics import get_auto_learning_rate
from ludwig.utils.llm_utils import create_text_streamer, TextStreamer
from ludwig.utils.misc_utils import (
    get_commit_hash,
    get_file_names,
    get_from_registry,
    get_output_directory,
    set_saved_weights_in_checkpoint_flag,
)
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.tokenizers import HFTokenizer
from ludwig.utils.trainer_utils import get_training_report
from ludwig.utils.types import DataFrame
from ludwig.utils.upload_utils import HuggingFaceHub

logger = logging.getLogger(__name__)


@PublicAPI
@dataclass
class EvaluationFrequency:
    """Represents the frequency of periodic evaluation of a metric during training. For example:

    "every epoch"
    frequency: 1, period: EPOCH

    "every 50 steps".
    frequency: 50, period: STEP
    """

    frequency: float = 1.0
    period: str = "epoch"  # One of "epoch" or "step".

    EPOCH: ClassVar[str] = "epoch"  # One epoch is a single pass through the training set.
    STEP: ClassVar[str] = "step"  # One step is training on one mini-batch.


@PublicAPI
@dataclass
class TrainingStats:
    """Training statistics for all splits (training, validation, test)."""

    training: dict[str, Any]
    validation: dict[str, Any]
    test: dict[str, Any]
    evaluation_frequency: EvaluationFrequency = dataclasses.field(default_factory=EvaluationFrequency)

    def keys(self) -> list[str]:
        return [TRAINING, VALIDATION, TEST]

    def __contains__(self, key: object) -> bool:
        return (
            (key == TRAINING and self.training)
            or (key == VALIDATION and self.validation)
            or (key == TEST and self.test)
        )

    def __getitem__(self, key: str) -> dict[str, Any]:
        return {TRAINING: self.training, VALIDATION: self.validation, TEST: self.test}[key]

    # Make TrainingStats a proper Mapping so dict(ts) and generic helpers like
    # ludwig.utils.numerical_test_utils.assert_all_finite treat it as a dict
    # rather than falling back to integer-index iteration (KeyError(0)).
    _KEYS = (TRAINING, VALIDATION, TEST)

    def keys(self):  # noqa: F811
        return self._KEYS

    def __iter__(self):
        return iter(self._KEYS)


@PublicAPI
@dataclass
class PreprocessedDataset:
    training_set: Dataset
    validation_set: Dataset
    test_set: Dataset
    training_set_metadata: TrainingSetMetadataDict

    def __iter__(self):
        import warnings

        warnings.warn(
            "Tuple unpacking of PreprocessedDataset is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self.training_set, self.validation_set, self.test_set, self.training_set_metadata))

    def __getitem__(self, index):
        import warnings

        warnings.warn(
            "Indexed access of PreprocessedDataset is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self.training_set, self.validation_set, self.test_set, self.training_set_metadata)[index]


@PublicAPI
@dataclass
class TrainingResults:
    train_stats: TrainingStats
    preprocessed_data: PreprocessedDataset
    output_directory: str

    def __iter__(self):
        import warnings

        warnings.warn(
            "Tuple unpacking of TrainingResults is deprecated. "
            "Use attribute access instead: result.train_stats, result.preprocessed_data, result.output_directory",
            DeprecationWarning,
            stacklevel=2,
        )
        return iter((self.train_stats, self.preprocessed_data, self.output_directory))

    def __getitem__(self, index):
        import warnings

        warnings.warn(
            "Indexed access of TrainingResults is deprecated. Use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return (self.train_stats, self.preprocessed_data, self.output_directory)[index]


@PublicAPI
class LudwigModel:
    """High-level interface to Ludwig's train / predict / evaluate / experiment pipelines.

    Example:
        Train a model::

            config = {...}
            model = LudwigModel(config)
            train_stats, _, _ = model.train(dataset=file_path)
            # or with a DataFrame:
            train_stats, _, _ = model.train(dataset=dataframe)

        Load a previously trained model and predict::

            model = LudwigModel.load(model_dir)
            predictions, output_dir = model.predict(dataset=file_path)
            # or:
            predictions, output_dir = model.predict(dataset=dataframe)

        Evaluate::

            eval_stats, _, _ = model.evaluate(dataset=file_path)
    """

    def __init__(
        self,
        config: str | dict,
        logging_level: int = logging.ERROR,
        backend: Backend | str | None = None,
        gpus: str | int | list[int] | None = None,
        gpu_memory_limit: float | None = None,
        allow_parallel_threads: bool = True,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Initialize a LudwigModel.

        Args:
            config: In-memory config dict or path to a YAML config file.
            logging_level: Log level sent to stderr (e.g., logging.INFO).
            backend: Backend instance or string name (e.g., "local", "ray") used for
                preprocessing and training.
            gpus: GPUs to use; same syntax as CUDA_VISIBLE_DEVICES.
            gpu_memory_limit: Maximum memory fraction [0, 1] allowed per GPU device.
            allow_parallel_threads: Allow Torch to use multi-threading for performance
                at the cost of determinism.
            callbacks: List of `ludwig.callbacks.Callback` objects that provide hooks
                into the Ludwig pipeline.
        """
        # check if config is a path or a dict
        if isinstance(config, str):  # assume path
            config_dict = load_yaml(config)
            self.config_fp = config
        else:
            config_dict = copy.deepcopy(config)
            self.config_fp = None  # type: ignore [assignment]

        self._user_config = upgrade_config_dict_to_latest_version(config_dict)

        # Initialize the config object
        self.config_obj = ModelConfig.from_dict(self._user_config)

        # setup logging
        self.set_logging_level(logging_level)

        # setup Backend
        self.backend = initialize_backend(backend or self._user_config.get("backend"))
        logger.info(f"Using backend: {self.backend.BACKEND_TYPE}")
        self.callbacks = callbacks if callbacks is not None else []

        # setup PyTorch env (GPU allocation, etc.)
        self.backend.initialize_pytorch(
            gpus=gpus, gpu_memory_limit=gpu_memory_limit, allow_parallel_threads=allow_parallel_threads
        )

        # setup model
        self.model = None
        self.training_set_metadata: dict[str, dict] | None = None

        # online training state
        self._online_trainer = None

        # Zero-shot LLM usage.
        if (
            self.config_obj.model_type == MODEL_LLM
            and self.config_obj.trainer.type == "none"
            # Category output features require a vocabulary. The LLM LudwigModel should be initialized with
            # model.train(dataset).
            and self.config_obj.output_features[0].type == "text"
        ):
            self._initialize_llm_for_zero_shot()

    def _get_or_create_model(
        self, config_obj: ModelConfig | None = None, random_seed: int = default_random_seed
    ) -> None:
        """Single entry point for model instantiation.

        Creates self.model from config_obj (or self.config_obj) if it hasn't been created yet. Safe to call multiple
        times — no-ops if model exists.
        """
        if self.model is not None:
            return
        cfg = config_obj or self.config_obj
        logger.info(f"Creating {cfg.model_type} model")
        self.model = LudwigModel.create_model(cfg, random_seed=random_seed)

    def _initialize_llm_for_zero_shot(self, random_seed: int = default_random_seed):
        """Initialize the LLM for zero-shot (NoneTrainer) inference only."""
        self._get_or_create_model(random_seed=random_seed)

        if self.model.model.device.type == "cpu" and torch.cuda.is_available():
            logger.warning(f"LLM was initialized on {self.model.model.device}. Moving to GPU for inference.")
            self.model.model.to(torch.device("cuda"))

    def train(
        self,
        dataset: str | dict | pd.DataFrame | None = None,
        training_set: str | dict | pd.DataFrame | Dataset | None = None,
        validation_set: str | dict | pd.DataFrame | Dataset | None = None,
        test_set: str | dict | pd.DataFrame | Dataset | None = None,
        training_set_metadata: str | dict | None = None,
        data_format: str | None = None,
        experiment_name: str = "api_experiment",
        model_name: str = "run",
        model_resume_path: str | None = None,
        skip_save_training_description: bool = False,
        skip_save_training_statistics: bool = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        skip_save_processed_input: bool = False,
        output_directory: str | None = "results",
        random_seed: int = default_random_seed,
        **kwargs,
    ) -> TrainingResults:
        """Train the model on the provided dataset.

        Results are saved to `[output_directory]/[experiment_name]_[model_name]_n`,
        where `n` increments to differentiate repeated runs.

        Args:
            dataset: Source containing the full dataset. If it has a split column (0=train,
                1=validation, 2=test) it is used for splitting; otherwise the dataset is
                split randomly. Mutually exclusive with `training_set`.
            training_set: Source containing training data only.
            validation_set: Source containing validation data only.
            test_set: Source containing test data only.
            training_set_metadata: Pre-computed metadata dict or path to a `.meta.json`
                file produced by a previous Ludwig run on the same dataset.
            data_format: Format hint for data sources. Inferred automatically when
                `None`. Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`,
                `'excel'`, `'feather'`, `'fwf'`, `'hdf5'`, `'html'`, `'json'`,
                `'jsonl'`, `'parquet'`, `'pickle'`, `'sas'`, `'spss'`, `'stata'`,
                `'tsv'`.
            experiment_name: Name used when creating the output directory.
            model_name: Name used when creating the output directory.
            model_resume_path: Resume training from this checkpoint directory.
                Config, optimizer state, and training statistics are all restored.
            skip_save_training_description: Skip saving the experiment description JSON.
            skip_save_training_statistics: Skip saving training statistics JSON.
            skip_save_model: Skip saving model weights after each improvement.
                The returned model will have end-of-training weights rather than
                best-validation weights, and the model cannot be reloaded later.
            skip_save_progress: Skip saving per-epoch checkpoints used for resuming.
            skip_save_log: Skip saving TensorBoard logs.
            skip_save_processed_input: Skip caching the preprocessed HDF5/JSON files.
            output_directory: Root directory for all saved outputs.
            random_seed: Seed for data splitting, weight initialization, and shuffling.
            **kwargs: Additional keyword arguments forwarded to preprocessing.

        Returns:
            A `TrainingResults` namedtuple with fields:
            - `training_set_metadata`: feature-level preprocessing metadata.
            - `preprocessed_data`: `(training_set, validation_set, test_set)` datasets.
            - `output_directory`: path where all outputs were saved.
        """
        # Only reset the metadata if the model has not been trained before
        if self.training_set_metadata:
            logger.warning(
                "This model has been trained before. Its architecture has been defined by the original training set "
                "(for example, the number of possible categorical outputs). The current training data will be mapped "
                "to this architecture. If you want to change the architecture of the model, please concatenate your "
                "new training data with the original and train a new model from scratch."
            )
            training_set_metadata = self.training_set_metadata

        if self._user_config.get(HYPEROPT):
            print_boxed("WARNING")
            logger.warning(HYPEROPT_WARNING)

        # setup directories and file names
        if model_resume_path is not None:
            if path_exists(model_resume_path):
                output_directory = model_resume_path
                if self.backend.is_coordinator():
                    logger.info(f"Model resume path '{model_resume_path}' exists, trying to resume training.")
            else:
                if self.backend.is_coordinator():
                    logger.info(
                        f"Model resume path '{model_resume_path}' does not exist, starting training from scratch"
                    )
                model_resume_path = None

        if model_resume_path is None:
            if self.backend.is_coordinator():
                output_directory = get_output_directory(output_directory, experiment_name, model_name)
            else:
                output_directory = None

        # if we are skipping all saving,
        # there is no need to create a directory that will remain empty
        should_create_output_directory = not (
            skip_save_training_description
            and skip_save_training_statistics
            and skip_save_model
            and skip_save_progress
            and skip_save_log
            and skip_save_processed_input
        )

        output_url = output_directory
        with upload_output_directory(output_directory) as (output_directory, upload_fn):
            train_callbacks = self.callbacks
            if upload_fn is not None:
                # Upload output files (checkpoints, etc.) to remote storage at the end of
                # each epoch and evaluation, in case of failure in the middle of training.
                class UploadOnEpochEndCallback(Callback):
                    def on_eval_end(self, trainer, progress_tracker, save_path):
                        upload_fn()

                    def on_epoch_end(self, trainer, progress_tracker, save_path):
                        upload_fn()

                train_callbacks = train_callbacks + [UploadOnEpochEndCallback()]

            description_fn = training_stats_fn = model_dir = None
            if self.backend.is_coordinator():
                if should_create_output_directory:
                    makedirs(output_directory, exist_ok=True)
                description_fn, training_stats_fn, model_dir = get_file_names(output_directory)

            if isinstance(training_set, Dataset) and training_set_metadata is not None:
                preprocessed_data = PreprocessedDataset(training_set, validation_set, test_set, training_set_metadata)
            else:
                # save description
                if self.backend.is_coordinator():
                    description = get_experiment_description(
                        self.config_obj.to_dict(),
                        dataset=dataset,
                        training_set=training_set,
                        validation_set=validation_set,
                        test_set=test_set,
                        training_set_metadata=training_set_metadata,
                        data_format=data_format,
                        backend=self.backend,
                        random_seed=random_seed,
                    )

                    if not skip_save_training_description:
                        save_json(description_fn, description)

                    # print description
                    experiment_description = [
                        ["Experiment name", experiment_name],
                        ["Model name", model_name],
                        ["Output directory", output_directory],
                    ]
                    for key, value in description.items():
                        if key != "config":  # Config is printed separately.
                            experiment_description.append([key, pformat(value, indent=4)])

                    if self.backend.is_coordinator():
                        print_boxed("EXPERIMENT DESCRIPTION")
                        logger.info(tabulate(experiment_description, tablefmt="fancy_grid"))

                        print_boxed("LUDWIG CONFIG")
                        logger.info("User-specified config (with upgrades):\n")
                        logger.info(pformat(self._user_config, indent=4))
                        logger.info(
                            "\nFull config saved to:\n"
                            f"{output_directory}/{experiment_name}/model/model_hyperparameters.json"
                        )

                preprocessed_data = self.preprocess(  # type: ignore[assignment]
                    dataset=dataset,
                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    training_set_metadata=training_set_metadata,
                    data_format=data_format,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    model_resume_path=model_resume_path,
                    skip_save_training_description=skip_save_training_description,
                    skip_save_training_statistics=skip_save_training_statistics,
                    skip_save_model=skip_save_model,
                    skip_save_progress=skip_save_progress,
                    skip_save_log=skip_save_log,
                    skip_save_processed_input=skip_save_processed_input,
                    output_directory=output_directory,
                    random_seed=random_seed,
                    **kwargs,
                )
                training_set = preprocessed_data.training_set
                validation_set = preprocessed_data.validation_set
                test_set = preprocessed_data.test_set
                training_set_metadata = preprocessed_data.training_set_metadata

            self.training_set_metadata = training_set_metadata

            if self.backend.is_coordinator():
                dataset_statistics = generate_dataset_statistics(training_set, validation_set, test_set)

                if not skip_save_model:
                    # save train set metadata
                    os.makedirs(model_dir, exist_ok=True)  # type: ignore[arg-type]
                    save_json(  # type: ignore[arg-type]
                        os.path.join(model_dir, TRAIN_SET_METADATA_FILE_NAME), training_set_metadata
                    )

                logger.info("\nDataset Statistics")
                logger.info(tabulate(dataset_statistics, headers="firstrow", tablefmt="fancy_grid"))

            for callback in self.callbacks:
                callback.on_train_init(
                    base_config=self._user_config,
                    experiment_directory=output_directory,
                    experiment_name=experiment_name,
                    model_name=model_name,
                    output_directory=output_directory,
                    resume_directory=model_resume_path,
                )

            # Build model if not provided
            # if it was provided it means it was already loaded
            if not self.model:
                if self.backend.is_coordinator():
                    print_boxed("MODEL")
                # update model config with metadata properties derived from training set
                update_config_with_metadata(self.config_obj, training_set_metadata)
                logger.info("Warnings and other logs:")
                self.model = LudwigModel.create_model(self.config_obj, random_seed=random_seed)
                # update config with properties determined during model instantiation
                update_config_with_model(self.config_obj, self.model)
                set_saved_weights_in_checkpoint_flag(self.config_obj)

            # auto tune learning rate
            if hasattr(self.config_obj.trainer, "learning_rate") and self.config_obj.trainer.learning_rate == AUTO:
                detected_learning_rate = get_auto_learning_rate(self.config_obj)
                self.config_obj.trainer.learning_rate = detected_learning_rate

            with self.backend.create_trainer(
                model=self.model,
                config=self.config_obj.trainer,
                resume=model_resume_path is not None,
                skip_save_model=skip_save_model,
                skip_save_progress=skip_save_progress,
                skip_save_log=skip_save_log,
                callbacks=train_callbacks,
                random_seed=random_seed,
            ) as trainer:
                # auto tune batch size
                self._tune_batch_size_and_grad_accum(trainer, training_set, random_seed=random_seed)

                if (
                    self.config_obj.model_type == "LLM"
                    and trainer.config.type == "none"
                    and self.config_obj.adapter is not None
                    and self.config_obj.adapter.pretrained_adapter_weights is not None
                ):
                    trainer.model.initialize_adapter()  # Load pre-trained adapter weights for inference only

                # train model
                if self.backend.is_coordinator():
                    print_boxed("TRAINING")
                    if not skip_save_model:
                        self.save_config(model_dir)

                for callback in self.callbacks:
                    callback.on_train_start(
                        model=self.model,
                        config=self.config_obj.to_dict(),
                        config_fp=self.config_fp,
                    )

                try:
                    train_stats = trainer.train(
                        training_set,
                        validation_set=validation_set,
                        test_set=test_set,
                        save_path=model_dir,
                    )
                    self.model, train_trainset_stats, train_valiset_stats, train_testset_stats = train_stats

                    # Calibrate output probabilities and save model (coordinator-only).
                    # Must run after training completes, before final model parameters are saved.
                    if self.backend.is_coordinator():
                        calibrator = Calibrator(
                            self.model,
                            self.backend,
                            batch_size=trainer.eval_batch_size,
                        )
                        self._run_calibration(calibrator, validation_set, training_set, skip_save_model, model_dir)

                    # Evaluation Frequency
                    if self.config_obj.model_type == MODEL_ECD and self.config_obj.trainer.steps_per_checkpoint:
                        evaluation_frequency = EvaluationFrequency(
                            self.config_obj.trainer.steps_per_checkpoint, EvaluationFrequency.STEP
                        )
                    elif self.config_obj.model_type == MODEL_ECD and self.config_obj.trainer.checkpoints_per_epoch:
                        evaluation_frequency = EvaluationFrequency(
                            1.0 / self.config_obj.trainer.checkpoints_per_epoch, EvaluationFrequency.EPOCH
                        )
                    else:
                        evaluation_frequency = EvaluationFrequency(1, EvaluationFrequency.EPOCH)

                    # Unpack train()'s return.
                    # The statistics are all nested dictionaries of TrainerMetrics: feature_name -> metric_name ->
                    # List[TrainerMetric], with one entry per training checkpoint, according to steps_per_checkpoint.
                    # We reduce the dictionary of TrainerMetrics to a simple list of floats for interfacing with Ray
                    # Tune.
                    train_stats = TrainingStats(
                        metric_utils.reduce_trainer_metrics_dict(train_trainset_stats),
                        metric_utils.reduce_trainer_metrics_dict(train_valiset_stats),
                        metric_utils.reduce_trainer_metrics_dict(train_testset_stats),
                        evaluation_frequency,
                    )

                    # save training statistics
                    if self.backend.is_coordinator():
                        if not skip_save_training_statistics:
                            save_json(training_stats_fn, train_stats)

                    # results of the model with highest validation test performance
                    if (
                        self.backend.is_coordinator()
                        and validation_set is not None
                        and not self.config_obj.trainer.skip_all_evaluation
                    ):
                        print_boxed("TRAINING REPORT")
                        training_report = get_training_report(
                            trainer.validation_field,
                            trainer.validation_metric,
                            test_set is not None,
                            train_valiset_stats,
                            train_testset_stats,
                        )
                        logger.info(tabulate(training_report, tablefmt="fancy_grid"))
                        logger.info(f"\nFinished: {experiment_name}_{model_name}")
                        logger.info(f"Saved to: {output_directory}")
                finally:
                    for callback in self.callbacks:
                        callback.on_train_end(output_directory)

                self.training_set_metadata = training_set_metadata

                if self.is_merge_and_unload_set():
                    # For an LLM model trained with a LoRA adapter, merge first, then save the full model.
                    self.model.merge_and_unload(progressbar=self.config_obj.adapter.postprocessor.progressbar)

                    if self.backend.is_coordinator() and not skip_save_model:
                        self.model.save_base_model(model_dir)
                elif self.backend.is_coordinator() and not skip_save_model:
                    self.model.save(model_dir)

                # Save model card alongside the model (always)
                if self.backend.is_coordinator() and not skip_save_model:
                    try:
                        from ludwig.utils.model_card import save_model_card

                        save_model_card(
                            output_directory=output_directory,
                            config=self.config_obj.to_dict(),
                            training_set_metadata=training_set_metadata,
                            train_stats=train_stats,
                            model_dir=model_dir,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate model card: {e}")
                        logger.debug(traceback.format_exc())

                # Save training report (always, alongside the model)
                if self.backend.is_coordinator() and not skip_save_model:
                    try:
                        from ludwig.utils.training_report import save_training_report

                        save_training_report(
                            output_directory=output_directory,
                            config=self.config_obj.to_dict(),
                            training_set_metadata=training_set_metadata,
                            train_stats=train_stats,
                            model_dir=model_dir,
                            random_seed=random_seed,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate training report: {e}")
                        logger.debug(traceback.format_exc())

                # Synchronize model weights between workers
                self.backend.sync_model(self.model)

                print_boxed("FINISHED")
                # `preprocessed_data` is a 4-tuple from the two construction sites above
                # (either built from pre-provided datasets or from self.preprocess()).
                # TrainingResults declares `preprocessed_data: PreprocessedDataset`, so
                # wrap the tuple before returning — downstream callers like
                # `experiment()` access attributes (.validation_set etc.) rather than
                # unpacking positionally.
                if isinstance(preprocessed_data, tuple):
                    preprocessed_data = PreprocessedDataset(*preprocessed_data)
                return TrainingResults(train_stats, preprocessed_data, output_url)

    def train_online(
        self,
        dataset: str | dict | pd.DataFrame,
        training_set_metadata: str | dict | None = None,
        data_format: str = "auto",
        random_seed: int = default_random_seed,
    ) -> None:
        """Train the model for one epoch on `dataset` (online / incremental learning).

        Args:
            dataset: Source containing the training data for this epoch.
            training_set_metadata: Pre-computed metadata from a prior run. When
                `None`, metadata is derived from the provided dataset.
            data_format: Format hint for the data source. Inferred when `'auto'`.
            random_seed: Seed for data splitting and parameter initialization.
        """
        training_set_metadata = training_set_metadata or self.training_set_metadata
        preprocessing_params = get_preprocessing_params(self.config_obj)

        with provision_preprocessing_workers(self.backend):
            training_dataset, _, _, training_set_metadata = preprocess_for_training(
                self.config_obj,
                training_set=dataset,
                training_set_metadata=training_set_metadata,
                data_format=data_format,
                skip_save_processed_input=True,
                preprocessing_params=preprocessing_params,
                backend=self.backend,
                random_seed=random_seed,
                callbacks=self.callbacks,
            )

        if not self.training_set_metadata:
            self.training_set_metadata = training_set_metadata

        if not self.model:
            update_config_with_metadata(self.config_obj, training_set_metadata)
            self.model = LudwigModel.create_model(self.config_obj, random_seed=random_seed)
            # update config with properties determined during model instantiation
            update_config_with_model(self.config_obj, self.model)
            set_saved_weights_in_checkpoint_flag(self.config_obj)

        if not self._online_trainer:
            self._online_trainer = self.backend.create_trainer(
                config=self.config_obj.trainer, model=self.model, random_seed=random_seed
            )

            self._tune_batch_size_and_grad_accum(self._online_trainer, dataset, random_seed=random_seed)

        self.model = self._online_trainer.train_online(training_dataset)

    def _run_calibration(
        self,
        calibrator: Calibrator,
        validation_set: Dataset | None,
        training_set: Dataset,
        skip_save_model: bool,
        model_dir: str,
    ) -> None:
        """Run post-training probability calibration and save the model.

        Must be called only on the coordinator node, after training completes and
        before the final model is saved.
        """
        if calibrator.calibration_enabled():
            if validation_set is None:
                logger.warning(
                    "Calibration uses validation set, but no validation split specified. "
                    "Will use training set for calibration. "
                    "Recommend providing a validation set when using calibration."
                )
                calibrator.train_calibration(training_set, TRAINING)
            elif len(validation_set) < MIN_DATASET_SPLIT_ROWS:
                logger.warning(
                    f"Validation set size ({len(validation_set)} rows) is too small for calibration. "
                    "Will use training set for calibration. "
                    f"Validation set must have at least {MIN_DATASET_SPLIT_ROWS} rows."
                )
                calibrator.train_calibration(training_set, TRAINING)
            else:
                calibrator.train_calibration(validation_set, VALIDATION)
        if not skip_save_model:
            self.model.save(model_dir)

    def _tune_batch_size_and_grad_accum(self, trainer, dataset, random_seed: int = default_random_seed):
        """Sets AUTO batch-size-related parameters based on the trainer, backend type, and number of workers.

        Batch-size related parameters that are set:
        - trainer.batch_size
        - trainer.eval_batch_size
        - trainer.gradient_accumulation_steps
        - trainer.effective_batch_size

        The final batch size selected may be non-deterministic even with a fixed random seed since throughput-based
        heuristics may be affected by resources used by other processes running on the machine.
        """
        if not self.config_obj.trainer.can_tune_batch_size():
            # Some model types don't have batch sizes to be tuned
            return

        # Render the batch size and gradient accumulation steps prior to batch size tuning. This is needed in the event
        # the effective_batch_size and gradient_accumulation_steps are set explicitly, but batch_size is AUTO. In this
        # case, we can infer the batch_size directly without tuning.
        num_workers = self.backend.num_training_workers
        self.config_obj.trainer.update_batch_size_grad_accum(num_workers)

        if self.config_obj.trainer.batch_size == AUTO:
            if self.backend.supports_batch_size_tuning():
                tuned_batch_size = trainer.tune_batch_size(
                    self.config_obj, dataset, random_seed=random_seed, tune_for_training=True
                )
            else:
                logger.warning(
                    f"Backend {self.backend.BACKEND_TYPE} does not support batch size tuning, "
                    f"using fallback training batch size {FALLBACK_BATCH_SIZE}."
                )
                tuned_batch_size = FALLBACK_BATCH_SIZE

            self.config_obj.trainer.batch_size = tuned_batch_size

            # Re-render the gradient_accumulation_steps to account for the explicit batch size.
            self.config_obj.trainer.update_batch_size_grad_accum(num_workers)

        if self.config_obj.trainer.eval_batch_size in {AUTO, None}:
            if self.backend.supports_batch_size_tuning():
                tuned_batch_size = trainer.tune_batch_size(
                    self.config_obj, dataset, random_seed=random_seed, tune_for_training=False
                )
            else:
                logger.warning(
                    f"Backend {self.backend.BACKEND_TYPE} does not support batch size tuning, "
                    f"using fallback eval batch size {FALLBACK_BATCH_SIZE}."
                )
                tuned_batch_size = FALLBACK_BATCH_SIZE

            self.config_obj.trainer.eval_batch_size = tuned_batch_size

        # Update trainer params separate to config params for backends with stateful trainers
        trainer.batch_size = self.config_obj.trainer.batch_size
        trainer.eval_batch_size = self.config_obj.trainer.eval_batch_size
        trainer.gradient_accumulation_steps = self.config_obj.trainer.gradient_accumulation_steps

    def save_dequantized_base_model(self, save_path: str) -> None:
        """Upscales quantized weights of a model to fp16 and saves the result in a specified folder.

        Args:
            save_path (str): The path to the folder where the upscaled model weights will be saved.

        Raises:
            ValueError:
                If the model type is not 'llm' or if quantization is not enabled or the number of bits is not 4 or 8.
            RuntimeError:
                If no GPU is available, as GPU is required for quantized models.

        Returns:
            None
        """
        if self.config_obj.model_type != MODEL_LLM:
            raise ValueError(
                f"Model type {self.config_obj.model_type} is not supported by this method. Only `llm` model type is "
                "supported."
            )

        if not self.config_obj.quantization:
            raise ValueError(
                "Quantization is not enabled in your Ludwig model config. "
                "To enable quantization, set `quantization` to `{'bits': 4}` or `{'bits': 8}` in your model config."
            )

        if self.config_obj.quantization.bits != 4:
            raise ValueError(
                "This method only works with quantized models with 4 bits. "
                "Support for 8-bit quantized models will be added in a future release."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for quantized models but no GPU found.")

        # Create the LLM model class instance with the loaded LLM if it hasn't been initialized yet.
        if not self.model:
            self.model = LudwigModel.create_model(self.config_obj)

        self.model.save_dequantized_base_model(save_path)

        logger.info(
            "If you want to upload this model to huggingface.co, run the following Python commands: \n"
            "from ludwig.utils.hf_utils import upload_folder_to_hfhub; \n"
            f"upload_folder_to_hfhub(repo_id='desired/huggingface/repo/name', folder_path='{save_path}')"
        )

    def generate(
        self,
        input_strings: str | list[str],
        generation_config: dict | None = None,
        streaming: bool | None = False,
        callbacks: list[Callback] | None = None,
    ) -> str | list[str]:
        """A simple generate() method that directly uses the underlying transformers library to generate text.

        Args:
            input_strings: Input text or list of texts to generate from.
            generation_config: Configuration for text generation.
            streaming: If True, enable streaming output.
            callbacks: Optional callbacks for this generate call.

        Returns:
            Union[str, List[str]]: Generated text or list of generated texts.
        """
        if self.config_obj.model_type != MODEL_LLM:
            raise ValueError(
                f"Model type {self.config_obj.model_type} is not supported by this method. Only `llm` model type is "
                "supported."
            )
        if not torch.cuda.is_available():
            # GPU is required for loading quantized models. See https://github.com/ludwig-ai/ludwig/issues/3695.
            raise ValueError(
                "A CUDA GPU is required for generate() with quantized LLMs, but none was detected.\n"
                "Either run on a GPU machine or disable quantization in your model config."
            )

        # Decoder-only models require left-padding for correct generation results (right-padding causes HF warnings).
        padding_side = "left" if not getattr(self.model.model.config, "is_encoder_decoder", False) else "right"
        tokenizer = HFTokenizer(self.config_obj.base_model, padding_side=padding_side)

        with self.model.use_generation_config(generation_config):
            start_time = time.time()
            tokenized_inputs = tokenizer.tokenizer(input_strings, return_tensors="pt", padding=True)
            input_ids = tokenized_inputs["input_ids"].to("cuda")
            attention_mask = tokenized_inputs["attention_mask"].to("cuda")

            if streaming:
                streamer = create_text_streamer(tokenizer.tokenizer)
                outputs = self._generate_streaming_outputs(input_strings, input_ids, attention_mask, streamer)
            else:
                outputs = self._generate_non_streaming_outputs(input_strings, input_ids, attention_mask)

            decoded_outputs = tokenizer.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            logger.info(f"Finished generating in: {(time.time() - start_time):.2f}s.")

            return decoded_outputs[0] if len(decoded_outputs) == 1 else decoded_outputs

    def _generate_streaming_outputs(
        self,
        input_strings: str | list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        streamer: TextStreamer,
    ) -> torch.Tensor:
        """Generate streaming outputs for the given input.

        Args:
            input_strings (Union[str, List[str]]): Input text or list of texts to generate from.
            input_ids (torch.Tensor): Tensor containing input IDs.
            attention_mask (torch.Tensor): Tensor containing attention masks.
            streamer (Union[TextStreamer, None]): Text streamer instance for streaming output.

        Returns:
            torch.Tensor: Concatenated tensor of generated outputs.
        """
        outputs = []
        input_strings = input_strings if isinstance(input_strings, list) else [input_strings]
        for i in range(len(input_ids)):
            with torch.no_grad():
                logger.debug(f"Input: {input_strings[i]}\n")
                # NOTE: self.model.model.generation_config is not used here because it is the default
                # generation config that the CausalLM was initialized with, rather than the one set within the
                # context manager.
                generated_output = self.model.model.generate(
                    input_ids=input_ids[i].unsqueeze(0),
                    attention_mask=attention_mask[i].unsqueeze(0),
                    generation_config=self.model.generation,
                    streamer=streamer,
                )
                logger.debug("----------------------")
                outputs.append(generated_output)
        return torch.cat(outputs, dim=0)

    def _generate_non_streaming_outputs(
        self,
        _input_strings: str | list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate non-streaming outputs for the given input.

        Args:
            _input_strings (Union[str, List[str]]): Unused input parameter.
            input_ids (torch.Tensor): Tensor containing input IDs.
            attention_mask (torch.Tensor): Tensor containing attention masks.
            streamer (Union[TextStreamer, None]): Text streamer instance for streaming output.

        Returns:
            torch.Tensor: Tensor of generated outputs.
        """
        with torch.no_grad():
            # NOTE: self.model.model.generation_config is not used here because it is the default
            # generation config that the CausalLM was initialized with, rather than the one set within the
            # context manager.
            return self.model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.model.generation,
            )

    def predict(
        self,
        dataset: str | dict | pd.DataFrame | None = None,
        data_format: str | None = None,
        split: str = FULL,
        batch_size: int = 128,
        generation_config: dict | None = None,
        skip_save_unprocessed_output: bool = True,
        skip_save_predictions: bool = True,
        output_directory: str = "results",
        return_type: type = pd.DataFrame,
        callbacks: list[Callback] | None = None,
        **kwargs,
    ) -> tuple[dict | pd.DataFrame, str]:
        """Make predictions from a trained model on the provided dataset.

        Args:
            dataset: Source containing the dataset to predict on.
            data_format: Format hint for the data source. Inferred automatically when `None`.
                Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
                `'fwf'`, `'hdf5'`, `'html'`, `'json'`, `'jsonl'`, `'parquet'`, `'pickle'`,
                `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
            split: Which split of the data to use when the dataset contains a split column.
                One of `'full'`, `'training'`, `'validation'`, `'test'`.
            batch_size: Number of rows per prediction batch.
            generation_config: LLM-only generation parameters. When `None`, the config used
                at training time is applied. Ignored for non-LLM models.
            skip_save_unprocessed_output: When `False`, raw numpy tensors are saved alongside
                the postprocessed CSV files. When `True` (default), only CSVs are written.
            skip_save_predictions: Skip writing prediction CSV files.
            output_directory: Root directory for saved prediction outputs.
            return_type: Format of the returned predictions (`pd.DataFrame` or `dict`).
            callbacks: Extra callbacks for this predict call; combined with any callbacks
                already registered to the model.
            **kwargs: Forwarded to the underlying predictor.

        Returns:
            A tuple `(predictions, output_directory)` where `predictions` is a
            `pd.DataFrame` (or `dict`) of model outputs and `output_directory` is
            the path where results were saved.
        """
        self._check_initialization()

        # preprocessing
        start_time = time.time()
        logger.debug(f"Preprocessing dataset for prediction (batch_size={batch_size})")
        dataset, _ = self._preprocess_for_prediction(
            dataset,
            data_format=data_format,
            split=split,
            include_outputs=False,
            callbacks=callbacks,
        )

        logger.debug(f"Running batch prediction (batch_size={batch_size})")
        with self.backend.create_predictor(self.model, batch_size=batch_size) as predictor:
            with self.model.use_generation_config(generation_config):
                predictions = predictor.batch_predict(
                    dataset,
                )

            if self.backend.is_coordinator():
                # if we are skipping all saving,
                # there is no need to create a directory that will remain empty
                should_create_exp_dir = not (skip_save_unprocessed_output and skip_save_predictions)
                if should_create_exp_dir:
                    makedirs(output_directory, exist_ok=True)

            logger.debug("Postprocessing predictions")
            postproc_predictions = postprocess(
                predictions,
                self.model.output_features,
                self.training_set_metadata,
                output_directory=output_directory,
                backend=self.backend,
                skip_save_unprocessed_output=skip_save_unprocessed_output or not self.backend.is_coordinator(),
            )
            converted_postproc_predictions = convert_predictions(
                postproc_predictions, self.model.output_features, return_type=return_type, backend=self.backend
            )
            if self.backend.is_coordinator():
                if not skip_save_predictions:
                    save_prediction_outputs(
                        postproc_predictions, self.model.output_features, output_directory, self.backend
                    )

                    logger.info(f"Saved to: {output_directory}")

            logger.info(f"Finished predicting in: {(time.time() - start_time):.2f}s.")
            return converted_postproc_predictions, output_directory

    def evaluate(
        self,
        dataset: str | dict | pd.DataFrame | None = None,
        data_format: str | None = None,
        split: str = FULL,
        batch_size: int | None = None,
        skip_save_unprocessed_output: bool = True,
        skip_save_predictions: bool = True,
        skip_save_eval_stats: bool = True,
        collect_predictions: bool = False,
        collect_overall_stats: bool = False,
        output_directory: str = "results",
        return_type: type = pd.DataFrame,
        **kwargs,
    ) -> tuple[dict, dict | pd.DataFrame, str]:
        """Evaluate a trained model and compute performance statistics.

        Args:
            dataset: Source containing the dataset to evaluate.
            data_format: Format hint for the data source. Inferred automatically when `None`.
                Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
                `'fwf'`, `'hdf5'`, `'html'`, `'json'`, `'jsonl'`, `'parquet'`, `'pickle'`,
                `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
            split: Which split of the data to use when the dataset contains a split column.
                One of `'full'`, `'training'`, `'validation'`, `'test'`.
            batch_size: Number of rows per evaluation batch. Defaults to `eval_batch_size`
                from the trainer config.
            skip_save_unprocessed_output: When `False`, raw numpy tensors are saved alongside
                postprocessed CSV files. When `True` (default), only CSVs are written.
            skip_save_predictions: Skip writing prediction CSV files.
            skip_save_eval_stats: Skip writing evaluation statistics JSON.
            collect_predictions: Collect and return postprocessed predictions.
            collect_overall_stats: Compute and include dataset-level aggregate metrics.
            output_directory: Root directory for saved evaluation outputs.
            return_type: Format for returned predictions (`pd.DataFrame` or `dict`).
            **kwargs: Forwarded to preprocessing.

        Returns:
            A tuple `(eval_stats, predictions, output_directory)` where `eval_stats` is a
            nested dict of feature → metric → value, `predictions` is a `pd.DataFrame` or
            `dict` of model outputs, and `output_directory` is the path where results were
            saved.
        """
        self._check_initialization()

        for callback in self.callbacks:
            callback.on_evaluation_start()

        # preprocessing
        logger.debug("Preprocessing dataset for evaluation")
        dataset, training_set_metadata = self._preprocess_for_prediction(
            dataset,
            data_format=data_format,
            split=split,
            include_outputs=True,
        )

        # Fallback to use eval_batch_size or batch_size if not provided
        if batch_size is None:
            # Requires dictionary getter since some trainer configs may not have a batch_size param
            trainer_dict = self.config_obj.trainer.to_dict()
            batch_size = trainer_dict.get(EVAL_BATCH_SIZE) or trainer_dict.get(BATCH_SIZE)
        if batch_size is None:
            raise ValueError(
                "batch_size not specified and no default found in trainer config. "
                "Set batch_size or eval_batch_size in your trainer config."
            )

        logger.debug(f"Running batch evaluation (batch_size={batch_size})")
        with self.backend.create_predictor(self.model, batch_size=batch_size) as predictor:
            eval_stats, predictions = predictor.batch_evaluation(
                dataset,
                collect_predictions=collect_predictions or collect_overall_stats,
            )

            # calculate the overall metrics
            if collect_overall_stats:
                dataset = dataset.to_df()

                overall_stats = calculate_overall_stats(
                    self.model.output_features, predictions, dataset, training_set_metadata
                )
                eval_stats = {
                    of_name: (
                        {**eval_stats[of_name], **overall_stats[of_name]}
                        # account for presence of 'combined' key
                        if of_name in overall_stats
                        else {**eval_stats[of_name]}
                    )
                    for of_name in eval_stats
                }

            if self.backend.is_coordinator():
                # if we are skipping all saving,
                # there is no need to create a directory that will remain empty
                should_create_exp_dir = not (
                    skip_save_unprocessed_output and skip_save_predictions and skip_save_eval_stats
                )
                if should_create_exp_dir:
                    makedirs(output_directory, exist_ok=True)

            if collect_predictions:
                logger.debug("Postprocessing predictions")
                postproc_predictions = postprocess(
                    predictions,
                    self.model.output_features,
                    self.training_set_metadata,
                    output_directory=output_directory,
                    backend=self.backend,
                    skip_save_unprocessed_output=skip_save_unprocessed_output or not self.backend.is_coordinator(),
                )
            else:
                postproc_predictions = predictions  # = {}

            if self.backend.is_coordinator():
                should_save_predictions = (
                    collect_predictions and postproc_predictions is not None and not skip_save_predictions
                )
                if should_save_predictions:
                    save_prediction_outputs(
                        postproc_predictions, self.model.output_features, output_directory, self.backend
                    )

                print_evaluation_stats(eval_stats)
                if not skip_save_eval_stats:
                    save_evaluation_stats(eval_stats, output_directory)

                if should_save_predictions or not skip_save_eval_stats:
                    logger.info(f"Saved to: {output_directory}")

            if collect_predictions:
                postproc_predictions = convert_predictions(
                    postproc_predictions, self.model.output_features, return_type=return_type, backend=self.backend
                )

            for callback in self.callbacks:
                callback.on_evaluation_end()

            return eval_stats, postproc_predictions, output_directory

    def forecast(
        self,
        dataset: DataFrame,
        data_format: str | None = None,
        horizon: int = 1,
        output_directory: str | None = None,
        output_format: str = "parquet",
        callbacks: list[Callback] | None = None,
    ) -> DataFrame:
        """Forecast `horizon` steps ahead using an iterative single-pass strategy.

        Preprocessing is performed once for the initial lookback window. Each subsequent horizon step slides the window
        by one position using incremental_time_delay_embedding, reducing preprocessing complexity from O(horizon ×
        window_size) to O(window_size + horizon).
        """
        self._check_initialization()

        # Load raw DataFrame once
        dataset, _, _, _ = load_dataset_uris(dataset, None, None, None, self.backend)
        if isinstance(dataset, CacheableDataset):
            dataset = dataset.unwrap()
        df = load_dataset(dataset, data_format=data_format, df_lib=self.backend.df_engine.df_lib)

        ts_input_features = [f for f in self.config_obj.input_features if f.type == TIMESERIES]
        ts_output_features = [f for f in self.config_obj.output_features if f.type == TIMESERIES]

        if not ts_input_features:
            raise ValueError("Forecasting requires at least one input feature of type `timeseries`.")

        if horizon <= 0:
            return_cols = [f.column for f in ts_output_features]
            return pd.DataFrame({col: pd.Series(dtype=float) for col in return_cols})

        max_window_size = max(f.preprocessing.window_size for f in ts_input_features)

        # Build a mapping from ts output column name → ts output feature config
        ts_output_by_col = {f.column: f for f in ts_output_features}

        # Step 1: Preprocess the initial lookback window once
        initial_df = df.tail(max_window_size)
        preprocessed, _ = self._preprocess_for_prediction(
            initial_df,
            include_outputs=False,
            callbacks=callbacks,
        )

        # Collect the last preprocessed embedding for each input feature.
        # Non-timeseries features stay constant; timeseries features are slid per step.
        # Keyed by proc_column of the model's input features.
        last_embeddings: dict[str, np.ndarray] = {}
        for i_feat in self.model.input_features.values():
            pc = i_feat.proc_column
            if pc in preprocessed.dataset:
                last_embeddings[pc] = preprocessed.dataset[pc][-1].copy()

        # Build a mapping: ts_input_feature.column → (proc_column, window_size, padding_value)
        ts_input_info: list[tuple[str, str, int, float]] = []
        for ts_feat in ts_input_features:
            i_feat = self.model.input_features.get(ts_feat.name)
            if i_feat is not None and i_feat.proc_column in last_embeddings:
                ts_input_info.append(
                    (
                        ts_feat.column,
                        i_feat.proc_column,
                        ts_feat.preprocessing.window_size,
                        ts_feat.preprocessing.padding_value,
                    )
                )

        # Step 2: Incremental prediction loop — O(horizon) steps, each O(1) preprocessing
        predicted_rows: list[pd.DataFrame] = []
        total_forecasted = 0

        with self.backend.create_predictor(self.model, batch_size=1) as predictor:
            while total_forecasted < horizon:
                # Build a single-sample batch from the last embeddings
                batch = {pc: emb[np.newaxis] for pc, emb in last_embeddings.items()}

                # Run model forward pass on one sample, then postprocess
                raw_preds = predictor.predict_single(batch)
                postproc_preds = postprocess(
                    raw_preds,
                    self.model.output_features,
                    self.training_set_metadata,
                    backend=self.backend,
                    skip_save_unprocessed_output=True,
                )

                # Extract predicted values for each timeseries output feature
                next_series: dict[str, pd.Series] = {}
                for feat in ts_output_features:
                    key = f"{feat.name}_predictions"
                    next_series[feat.column] = pd.Series(postproc_preds[key].iloc[0])

                next_preds = pd.DataFrame(next_series)
                predicted_rows.append(next_preds)
                total_forecasted += len(next_preds)

                # Step 3: Update embeddings incrementally for the next step.
                # For each timeseries input feature, slide the window by one position.
                for ts_col, proc_col, window_size, padding_value in ts_input_info:
                    # Use the predicted value if this ts input is also an output, else padding_value
                    # (matches the NaN-fill behavior of the original full-reprocessing path).
                    new_val = float(next_preds[ts_col].iloc[-1]) if ts_col in ts_output_by_col else padding_value
                    last_embeddings[proc_col] = incremental_time_delay_embedding(
                        new_val, last_embeddings[proc_col], window_size, padding_value
                    )

        results_df = pd.concat(predicted_rows, ignore_index=True).head(horizon)
        return_cols = [f.column for f in ts_output_features]
        results_df = results_df[return_cols]

        if output_directory is not None:
            if self.backend.is_coordinator():
                if output_format == "parquet":
                    output_path = os.path.join(output_directory, "forecast.parquet")
                    results_df.to_parquet(output_path)
                elif output_format == "csv":
                    output_path = os.path.join(output_directory, "forecast.csv")
                    results_df.to_csv(output_path)
                else:
                    raise ValueError(f"`output_format` {output_format} not supported. Must be one of [parquet, csv]")
                logger.info(f"Saved to: {output_path}")

        return results_df

    def experiment(
        self,
        dataset: str | dict | pd.DataFrame | None = None,
        training_set: str | dict | pd.DataFrame | None = None,
        validation_set: str | dict | pd.DataFrame | None = None,
        test_set: str | dict | pd.DataFrame | None = None,
        training_set_metadata: str | dict | None = None,
        data_format: str | None = None,
        experiment_name: str = "experiment",
        model_name: str = "run",
        model_resume_path: str | None = None,
        eval_split: str = TEST,
        skip_save_training_description: bool = False,
        skip_save_training_statistics: bool = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        skip_save_processed_input: bool = False,
        skip_save_unprocessed_output: bool = False,
        skip_save_predictions: bool = False,
        skip_save_eval_stats: bool = False,
        skip_collect_predictions: bool = False,
        skip_collect_overall_stats: bool = False,
        output_directory: str = "results",
        random_seed: int = default_random_seed,
        **kwargs,
    ) -> tuple[dict | None, TrainingStats, PreprocessedDataset, str]:
        """Train a model and immediately evaluate it on a held-out split.

        Combines `train()` and `evaluate()` in one call. Saves the model,
        training statistics, and evaluation results to `output_directory`.

        Args:
            dataset: Source containing the full dataset. Mutually exclusive with
                `training_set` / `validation_set` / `test_set`.
            training_set: Source containing training data only.
            validation_set: Source containing validation data only.
            test_set: Source containing test data only.
            training_set_metadata: Pre-computed metadata dict or path to a `.meta.json`
                file from a prior Ludwig run on the same dataset.
            data_format: Format hint for data sources. Inferred automatically when `None`.
                Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
                `'fwf'`, `'hdf5'`, `'html'`, `'json'`, `'jsonl'`, `'parquet'`, `'pickle'`,
                `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
            experiment_name: Name used when creating the output directory.
            model_name: Name used when creating the output directory.
            model_resume_path: Resume training from this checkpoint directory.
            eval_split: Which split to evaluate after training. One of `'training'`,
                `'validation'`, `'test'`.
            skip_save_training_description: Skip saving the experiment description JSON.
            skip_save_training_statistics: Skip saving training statistics JSON.
            skip_save_model: Skip saving model weights after each improvement.
            skip_save_progress: Skip saving per-epoch checkpoints for resuming.
            skip_save_log: Skip saving TensorBoard logs.
            skip_save_processed_input: Skip caching the preprocessed HDF5/JSON files.
            skip_save_unprocessed_output: Skip saving raw numpy prediction tensors.
            skip_save_predictions: Skip writing prediction CSV files.
            skip_save_eval_stats: Skip writing evaluation statistics JSON.
            skip_collect_predictions: Do not collect postprocessed predictions.
            skip_collect_overall_stats: Do not compute dataset-level aggregate metrics.
            output_directory: Root directory for all saved outputs.
            random_seed: Seed for weight initialization, data splitting, and shuffling.
            **kwargs: Forwarded to preprocessing.

        Returns:
            A tuple `(eval_stats, train_stats, preprocessed_data, output_directory)` where
            `eval_stats` is performance metrics on the eval split (or `None` if eval was
            skipped), `train_stats` is per-epoch training metrics, `preprocessed_data`
            holds the three split datasets, and `output_directory` is where results were
            saved.
        """
        if self._user_config.get(HYPEROPT):
            print_boxed("WARNING")
            logger.warning(HYPEROPT_WARNING)

        train_result = self.train(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            experiment_name=experiment_name,
            model_name=model_name,
            model_resume_path=model_resume_path,
            skip_save_training_description=skip_save_training_description,
            skip_save_training_statistics=skip_save_training_statistics,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            skip_save_unprocessed_output=skip_save_unprocessed_output,
            output_directory=output_directory,
            random_seed=random_seed,
        )
        train_stats = train_result.train_stats
        preprocessed_data = train_result.preprocessed_data
        output_directory = train_result.output_directory

        eval_set = preprocessed_data.validation_set
        if eval_split == TRAINING:
            eval_set = preprocessed_data.training_set
        elif eval_split == VALIDATION:
            eval_set = preprocessed_data.validation_set
        elif eval_split == TEST:
            eval_set = preprocessed_data.test_set
        else:
            logger.warning(f"Eval split {eval_split} not supported. Using validation set instead")

        if eval_set is not None:
            trainer_dict = self.config_obj.trainer.to_dict()
            batch_size = trainer_dict.get(EVAL_BATCH_SIZE, trainer_dict.get(BATCH_SIZE, None))

            # predict
            try:
                eval_stats, _, _ = self.evaluate(
                    eval_set,
                    data_format=data_format,
                    batch_size=batch_size,
                    output_directory=output_directory,
                    skip_save_unprocessed_output=skip_save_unprocessed_output,
                    skip_save_predictions=skip_save_predictions,
                    skip_save_eval_stats=skip_save_eval_stats,
                    collect_predictions=not skip_collect_predictions,
                    collect_overall_stats=not skip_collect_overall_stats,
                    return_type="dict",
                )
            except NotImplementedError:
                logger.warning(
                    "Skipping evaluation as the necessary methods are not "
                    "supported. Full exception below:\n"
                    f"{traceback.format_exc()}"
                )
                eval_stats = None
        else:
            logger.warning(f"The evaluation set {eval_set} was not provided. Skipping evaluation")
            eval_stats = None

        return eval_stats, train_stats, preprocessed_data, output_directory

    def collect_weights(self, tensor_names: list[str] | None = None, **kwargs) -> list:
        """Return the named tensors (weight matrices) from the trained model.

        Args:
            tensor_names: Names of tensors to retrieve. When `None`, all tensors
                are returned.
            **kwargs: Unused; accepted for forward-compatibility.

        Returns:
            List of `(name, tensor)` tuples.
        """
        self._check_initialization()
        collected_tensors = self.model.collect_weights(tensor_names)
        return collected_tensors

    def collect_activations(
        self,
        layer_names: list[str],
        dataset: str | dict[str, list] | pd.DataFrame,
        data_format: str | None = None,
        split: str = FULL,
        batch_size: int = 128,
        **kwargs,
    ) -> list:
        """Collect intermediate-layer activations for the given dataset.

        Args:
            layer_names: Names of layers in the model to collect activations from.
            dataset: Source containing the data to run through the model.
            data_format: Format hint for the data source. Inferred when `None`.
            split: Which data split to use when the dataset has a split column.
                One of `'full'`, `'training'`, `'validation'`, `'test'`.
            batch_size: Number of rows per inference batch.
            **kwargs: Unused; accepted for forward-compatibility.

        Returns:
            List of activation tensors, one per layer name.
        """
        self._check_initialization()

        # preprocessing
        logger.debug("Preprocessing dataset for activation collection")
        dataset, training_set_metadata = self._preprocess_for_prediction(
            dataset,
            data_format=data_format,
            split=split,
            include_outputs=False,
        )

        logger.debug(f"Collecting activations for layers: {layer_names} (batch_size={batch_size})")
        with self.backend.create_predictor(self.model, batch_size=batch_size) as predictor:
            activations = predictor.batch_collect_activations(
                layer_names,
                dataset,
            )

            return activations

    def preprocess(
        self,
        dataset: str | dict | pd.DataFrame | None = None,
        training_set: str | dict | pd.DataFrame | None = None,
        validation_set: str | dict | pd.DataFrame | None = None,
        test_set: str | dict | pd.DataFrame | None = None,
        training_set_metadata: str | dict | None = None,
        data_format: str | None = None,
        skip_save_processed_input: bool = True,
        random_seed: int = default_random_seed,
        **kwargs,
    ) -> PreprocessedDataset:
        """Preprocess a dataset and return it split into training / validation / test sets.

        Args:
            dataset: Source containing the full dataset. Mutually exclusive with
                `training_set` / `validation_set` / `test_set`.
            training_set: Source containing training data only.
            validation_set: Source containing validation data only.
            test_set: Source containing test data only.
            training_set_metadata: Pre-computed metadata dict or `.meta.json` path
                from a prior Ludwig run on the same dataset.
            data_format: Format hint for data sources. Inferred when `None`.
                Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`,
                `'feather'`, `'fwf'`, `'hdf5'`, `'html'`, `'json'`, `'jsonl'`,
                `'parquet'`, `'pickle'`, `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
            skip_save_processed_input: Skip caching the preprocessed HDF5/JSON files.
            random_seed: Seed for data splitting and shuffling.
            **kwargs: Forwarded to the underlying preprocessing function.

        Returns:
            A `PreprocessedDataset` namedtuple with fields `training_set`,
            `validation_set`, `test_set`, and `training_set_metadata`.

        Raises:
            RuntimeError: If preprocessing fails (e.g., empty training set after
                filtering, or lazy loading incompatible with RayBackend).
        """
        print_boxed("PREPROCESSING")

        for callback in self.callbacks:
            callback.on_preprocess_start(self.config_obj.to_dict())

        preprocessing_params = get_preprocessing_params(self.config_obj)

        proc_training_set = proc_validation_set = proc_test_set = None
        try:
            with provision_preprocessing_workers(self.backend):
                preprocessed_data = preprocess_for_training(
                    self.config_obj,
                    dataset=dataset,
                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    training_set_metadata=training_set_metadata,
                    data_format=data_format,
                    skip_save_processed_input=skip_save_processed_input,
                    preprocessing_params=preprocessing_params,
                    backend=self.backend,
                    random_seed=random_seed,
                    callbacks=self.callbacks,
                )

            proc_training_set, proc_validation_set, proc_test_set, training_set_metadata = preprocessed_data

            return PreprocessedDataset(proc_training_set, proc_validation_set, proc_test_set, training_set_metadata)
        except Exception:
            logger.debug(traceback.format_exc())
            raise
        finally:
            for callback in self.callbacks:
                callback.on_preprocess_end(proc_training_set, proc_validation_set, proc_test_set, training_set_metadata)

    @staticmethod
    def load(
        model_dir: str,
        logging_level: int = logging.ERROR,
        backend: Backend | str | None = None,
        gpus: str | int | list[int] | None = None,
        gpu_memory_limit: float | None = None,
        allow_parallel_threads: bool = True,
        callbacks: list[Callback] | None = None,
        from_checkpoint: bool = False,
    ) -> "LudwigModel":  # return is an instance of ludwig.api.LudwigModel class
        """Load a previously trained LudwigModel from disk.

        Args:
            model_dir: Path to the saved model directory (typically
                `results/<experiment>/<model>/model/`).
            logging_level: Log level sent to stderr (e.g., `logging.INFO`).
            backend: Backend instance or string name used for preprocessing.
            gpus: GPUs to use; same syntax as CUDA_VISIBLE_DEVICES.
            gpu_memory_limit: Maximum memory fraction [0, 1] allowed per GPU.
            allow_parallel_threads: Allow Torch multi-threading for performance
                at the cost of determinism.
            callbacks: List of `Callback` objects providing hooks into the pipeline.
            from_checkpoint: When `True`, load from the latest training checkpoint
                in `training_checkpoints/` instead of the final model weights.

        Returns:
            A fully initialized `LudwigModel` ready for inference.

        Example::

            model = LudwigModel.load("results/experiment/run/model")
            predictions, _ = model.predict(dataset=df)
        """
        # Initialize PyTorch before calling `broadcast()` to prevent initializing
        # Torch with default parameters
        backend_param = backend
        backend = initialize_backend(backend)
        backend.initialize_pytorch(
            gpus=gpus, gpu_memory_limit=gpu_memory_limit, allow_parallel_threads=allow_parallel_threads
        )

        logger.info(f"Loading model from {model_dir}")
        config = backend.broadcast_return(lambda: load_json(os.path.join(model_dir, MODEL_HYPERPARAMETERS_FILE_NAME)))

        # Upgrades deprecated fields and adds new required fields in case the config loaded from disk is old.
        config_obj = ModelConfig.from_dict(config)

        # Ensure that the original backend is used if it was specified in the config and user requests it
        if backend_param is None and "backend" in config:
            # Reset backend from config
            backend = initialize_backend(config.get("backend"))

        # initialize model
        ludwig_model = LudwigModel(
            config_obj.to_dict(),
            logging_level=logging_level,
            backend=backend,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
        )

        # generate model from config
        set_saved_weights_in_checkpoint_flag(config_obj)
        ludwig_model._get_or_create_model(config_obj)

        # load model weights
        logger.info(f"Loading model weights from {model_dir}")
        ludwig_model.load_weights(model_dir, from_checkpoint)

        # If merge_and_unload was NOT performed before saving (i.e., adapter weights exist),
        # we need to merge them now for inference.
        if ludwig_model.is_merge_and_unload_set():
            weights_save_path = os.path.join(model_dir, MODEL_WEIGHTS_FILE_NAME)
            adapter_config_path = os.path.join(weights_save_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                ludwig_model.model.merge_and_unload(progressbar=config_obj.adapter.postprocessor.progressbar)

        # load train set metadata
        ludwig_model.training_set_metadata = backend.broadcast_return(
            lambda: load_metadata(os.path.join(model_dir, TRAIN_SET_METADATA_FILE_NAME))
        )

        return ludwig_model

    def load_weights(
        self,
        model_dir: str,
        from_checkpoint: bool = False,
    ) -> None:
        """Load model weights from a saved model directory.

        Args:
            model_dir: Path to the saved model directory.
            from_checkpoint: When `True`, load from the latest training checkpoint
                instead of the final model weights.
        """
        if self.backend.is_coordinator():
            if from_checkpoint:
                with self.backend.create_trainer(
                    model=self.model,
                    config=self.config_obj.trainer,
                ) as trainer:
                    checkpoint = trainer.create_checkpoint_handle()
                    training_checkpoints_path = os.path.join(model_dir, TRAINING_CHECKPOINTS_DIR_PATH)
                    trainer.resume_weights_and_optimizer(training_checkpoints_path, checkpoint)
            else:
                self.model.load(model_dir)

        self.backend.sync_model(self.model)

    def save(self, save_path: str) -> None:
        """Save the model config, weights, and training metadata to `save_path`.

        Args:
            save_path: Directory where the model will be saved. Created if it
                does not exist. Contains `model_hyperparameters.json`, weight
                files, and `training_set_metadata.json`.
        """
        self._check_initialization()

        # save config
        self.save_config(save_path)

        # save model weights
        self.model.save(save_path)

        # save training set metadata
        training_set_metadata_path = os.path.join(save_path, TRAIN_SET_METADATA_FILE_NAME)
        save_json(training_set_metadata_path, self.training_set_metadata)

    @staticmethod
    def upload_to_hf_hub(
        repo_id: str,
        model_path: str,
        repo_type: str = "model",
        private: bool = False,
        commit_message: str = "Upload trained [Ludwig](https://ludwig.ai/latest/) model weights",
        commit_description: str | None = None,
    ) -> bool:
        """Uploads trained model artifacts to the HuggingFace Hub.

        Args:
            repo_id: A namespace (user or an organization) and a repo name separated by a `/`.
            model_path: The path of the saved model. This is either (a) the folder where the 'model_weights'
                folder and the 'model_hyperparameters.json' file are stored, or (b) the parent of that folder.
            private: Whether the model repo should be private. Defaults to False.
            repo_type: Set to `"dataset"` or `"space"` if uploading to a dataset or space, `None` or `"model"`
                if uploading to a model. Default is `None`.
            commit_message: The summary / title / first line of the generated commit.
            commit_description: The description of the generated commit.

        Returns:
            True for success, False for failure.
        """
        if model_weights_exist(os.path.join(model_path, MODEL_FILE_NAME)) and os.path.exists(
            os.path.join(model_path, MODEL_FILE_NAME, MODEL_HYPERPARAMETERS_FILE_NAME)
        ):
            experiment_path = model_path
        elif model_weights_exist(model_path) and os.path.exists(
            os.path.join(model_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        ):
            experiment_path = os.path.dirname(model_path)
        else:
            raise ValueError(
                f"Can't find model weights and '{MODEL_HYPERPARAMETERS_FILE_NAME}' either at "
                f"'{model_path}' or at '{model_path}/model'"
            )
        model_service = get_upload_registry()["hf_hub"]
        hub: HuggingFaceHub = model_service()
        hub.login()
        upload_status: bool = hub.upload(
            repo_id=repo_id,
            model_path=experiment_path,
            repo_type=repo_type,
            private=private,
            commit_message=commit_message,
            commit_description=commit_description,
        )
        return upload_status

    def save_config(self, save_path: str) -> None:
        """Save config to specified location.

        Args:
            save_path: filepath string to save config as a JSON file.
        """
        os.makedirs(save_path, exist_ok=True)
        model_hyperparameters_path = os.path.join(save_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        save_json(model_hyperparameters_path, self.config_obj.to_dict())

    def export_model(self, save_path: str, format: str = "safetensors", sample_input: dict | None = None) -> None:
        """Export the model in various formats.

        Args:
            save_path: Directory to save the exported model.
            format: Export format. One of "safetensors", "torch_export", "onnx".
            sample_input: Example input for tracing (required for torch_export and onnx).
        """
        from ludwig.utils.model_export import ModelExporter

        exporter = ModelExporter(self.model)

        if format == "safetensors":
            return exporter.export_safetensors(save_path)
        elif format == "torch_export":
            return exporter.export_torch(save_path, sample_input)
        elif format == "onnx":
            return exporter.export_onnx(save_path, sample_input)
        else:
            raise ValueError(f"Unknown export format: {format}. Options: safetensors, torch_export, onnx")

    def _preprocess_for_prediction(
        self,
        dataset: str | dict | pd.DataFrame | Dataset,
        data_format: str | None = None,
        split: str | None = None,
        include_outputs: bool = False,
        callbacks: list | None = None,
    ):
        """Shared preprocessing wrapper for predict, evaluate, and collect_activations."""
        return preprocess_for_prediction(
            self.config_obj,
            dataset=dataset,
            training_set_metadata=self.training_set_metadata,
            data_format=data_format,
            split=split,
            include_outputs=include_outputs,
            backend=self.backend,
            callbacks=self.callbacks + (callbacks or []),
        )

    def _check_initialization(self):
        missing = []
        if self.model is None:
            missing.append("model")
        if self._user_config is None:
            missing.append("config")
        if self.training_set_metadata is None:
            missing.append("training_set_metadata")
        if missing:
            raise ValueError(
                f"Model is not initialized (missing: {', '.join(missing)}). "
                "Call train() or load() before predict/evaluate."
            )

    def free_gpu_memory(self) -> None:
        """Manually moves the model to CPU to force GPU memory to be freed.

        For more context: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/35
        """
        if torch.cuda.is_available():
            self.model.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()

    @staticmethod
    def create_model(config_obj: ModelConfig | dict, random_seed: int = default_random_seed) -> BaseModel:
        """Instantiates BaseModel object.

        Args:
            config_obj: Ludwig config object.
            random_seed: Random seed used for weights initialization, splits and any other random function.

        Returns:
            Instance of the Ludwig model object.
        """
        if isinstance(config_obj, dict):
            config_obj = ModelConfig.from_dict(config_obj)
        model_type = get_from_registry(config_obj.model_type, model_type_registry)
        return model_type(config_obj, random_seed=random_seed)

    @staticmethod
    def set_logging_level(logging_level: int) -> None:
        """Sets level for log messages.

        Args:
            logging_level: Set/Update the logging level. Use logging constants like `logging.DEBUG`,
                `logging.INFO` and `logging.ERROR`.
        """
        logging.getLogger("ludwig").setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)
        else:
            set_disable_progressbar(False)

    @property
    def config(self) -> ModelConfigDict:
        """Returns the fully-rendered config of this model including default values."""
        return self.config_obj.to_dict()

    @config.setter
    def config(self, user_config: ModelConfigDict):
        """Updates the config of this model.

        WARNING: this can have unexpected results on an already trained model.
        """
        self._user_config = user_config
        self.config_obj = ModelConfig.from_dict(self._user_config)

    def is_merge_and_unload_set(self) -> bool:
        """Return True if this model is an LLM configured to merge_and_unload QLoRA adapter weights."""
        # TODO: In the future, it may be possible to move up the model type check into the BaseModel class.
        return self.config_obj.model_type == MODEL_LLM and self.model.is_merge_and_unload_set()


@PublicAPI
def kfold_cross_validate(
    num_folds: int,
    config: dict | str,
    dataset: str | None = None,
    data_format: str | None = None,
    skip_save_training_description: bool = False,
    skip_save_training_statistics: bool = False,
    skip_save_model: bool = False,
    skip_save_progress: bool = False,
    skip_save_log: bool = False,
    skip_save_processed_input: bool = False,
    skip_save_predictions: bool = False,
    skip_save_eval_stats: bool = False,
    skip_collect_predictions: bool = False,
    skip_collect_overall_stats: bool = False,
    output_directory: str = "results",
    random_seed: int = default_random_seed,
    gpus: str | int | list[int] | None = None,
    gpu_memory_limit: float | None = None,
    allow_parallel_threads: bool = True,
    backend: Backend | str | None = None,
    logging_level: int = logging.INFO,
    **kwargs,
) -> tuple[dict, dict]:
    """Perform k-fold cross-validation and return aggregated metrics.

    Args:
        num_folds: Number of folds for cross-validation.
        config: Model config dict or path to a YAML config file.
        dataset: Source containing the full dataset. Note: `'hdf5'` format is
            not supported for k-fold cross-validation.
        data_format: Format hint for the data source. Inferred automatically when
            `None`. Valid values: `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`,
            `'feather'`, `'fwf'`, `'html'`, `'json'`, `'jsonl'`, `'parquet'`,
            `'pickle'`, `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
        skip_save_training_description: Skip saving the experiment description JSON.
        skip_save_training_statistics: Skip saving training statistics JSON.
        skip_save_model: Skip saving model weights after each improvement.
        skip_save_progress: Skip saving per-epoch checkpoints for resuming.
        skip_save_log: Skip saving TensorBoard logs.
        skip_save_processed_input: Skip caching preprocessed HDF5/JSON files.
        skip_save_predictions: Skip writing prediction CSV files.
        skip_save_eval_stats: Skip writing evaluation statistics JSON.
        skip_collect_predictions: Do not collect postprocessed predictions.
        skip_collect_overall_stats: Do not compute dataset-level aggregate metrics.
        output_directory: Root directory for saved outputs.
        random_seed: Seed for weight initialization, data splitting, and shuffling.
        gpus: GPUs to use; same syntax as CUDA_VISIBLE_DEVICES.
        gpu_memory_limit: Maximum memory fraction [0, 1] allowed per GPU device.
        allow_parallel_threads: Allow Torch multi-threading at the cost of determinism.
        backend: Backend instance or string name for preprocessing and training.
        logging_level: Log level sent to stderr.
        **kwargs: Forwarded to each fold's `experiment()` call.

    Returns:
        A tuple `(kfold_cv_statistics, kfold_split_indices)` where
        `kfold_cv_statistics` maps fold name → training + eval metrics, and
        `kfold_split_indices` maps fold name → training/test index arrays.
    """
    # if config is a path, convert to dictionary
    if isinstance(config, str):  # assume path
        config = load_yaml(config)
    backend = initialize_backend(backend or config.get("backend"))

    # check for k_fold
    if num_folds is None:
        raise ValueError("k_fold parameter must be specified")

    logger.info(f"starting {num_folds:d}-fold cross validation")

    # create output_directory if not available
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # prepare data for k-fold processing
    # use Ludwig's utility to facilitate creating a dataframe
    # that is used as the basis for creating folds

    dataset, _, _, _ = load_dataset_uris(dataset, None, None, None, backend)

    # determine data format of provided dataset
    if not data_format or data_format == "auto":
        data_format = figure_data_format(dataset)

    data_df = load_dataset(dataset, data_format=data_format, df_lib=backend.df_engine.df_lib)

    kfold_cv_stats = {}
    kfold_split_indices = {}

    for train_indices, test_indices, fold_num in generate_kfold_splits(data_df, num_folds, random_seed):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            curr_train_df = data_df.iloc[train_indices]
            curr_test_df = data_df.iloc[test_indices]

            kfold_split_indices["fold_" + str(fold_num)] = {
                "training_indices": train_indices,
                "test_indices": test_indices,
            }

            # train and validate model on this fold
            logger.info(f"training on fold {fold_num:d}")

            model = LudwigModel(
                config=config,
                logging_level=logging_level,
                backend=backend,
                gpus=gpus,
                gpu_memory_limit=gpu_memory_limit,
                allow_parallel_threads=allow_parallel_threads,
            )
            eval_stats, train_stats, preprocessed_data, output_directory = model.experiment(
                training_set=curr_train_df,
                test_set=curr_test_df,
                experiment_name="cross_validation",
                model_name="fold_" + str(fold_num),
                skip_save_training_description=skip_save_training_description,
                skip_save_training_statistics=skip_save_training_statistics,
                skip_save_model=skip_save_model,
                skip_save_progress=skip_save_progress,
                skip_save_log=skip_save_log,
                skip_save_processed_input=skip_save_processed_input,
                skip_save_predictions=skip_save_predictions,
                skip_save_eval_stats=skip_save_eval_stats,
                skip_collect_predictions=skip_collect_predictions,
                skip_collect_overall_stats=skip_collect_overall_stats,
                output_directory=os.path.join(temp_dir_name, "results"),
                random_seed=random_seed,
            )

            # augment the training statistics with scoring metric from
            # the hold out fold
            if dataclasses.is_dataclass(train_stats):
                train_stats_dict = dataclasses.asdict(train_stats)
            elif hasattr(train_stats, "to_dict"):
                train_stats_dict = train_stats.to_dict()
            else:
                train_stats_dict = vars(train_stats)
            train_stats_dict["fold_eval_stats"] = eval_stats

            # collect training statistics for this fold
            kfold_cv_stats["fold_" + str(fold_num)] = train_stats_dict

    # consolidate raw fold metrics across all folds
    raw_kfold_stats = {}
    for fold_name in kfold_cv_stats:
        curr_fold_eval_stats = kfold_cv_stats[fold_name]["fold_eval_stats"]
        for of_name in curr_fold_eval_stats:
            if of_name not in raw_kfold_stats:
                raw_kfold_stats[of_name] = {}
            fold_eval_stats_of = curr_fold_eval_stats[of_name]

            for metric in fold_eval_stats_of:
                if metric not in {
                    "predictions",
                    "probabilities",
                    "confusion_matrix",
                    "overall_stats",
                    "per_class_stats",
                    "roc_curve",
                    "precision_recall_curve",
                }:
                    if metric not in raw_kfold_stats[of_name]:
                        raw_kfold_stats[of_name][metric] = []
                    raw_kfold_stats[of_name][metric].append(fold_eval_stats_of[metric])

    # calculate overall kfold statistics
    overall_kfold_stats = {}
    for of_name in raw_kfold_stats:
        overall_kfold_stats[of_name] = {}
        for metric in raw_kfold_stats[of_name]:
            mean = np.mean(raw_kfold_stats[of_name][metric])
            std = np.std(raw_kfold_stats[of_name][metric])
            overall_kfold_stats[of_name][metric + "_mean"] = mean
            overall_kfold_stats[of_name][metric + "_std"] = std

    kfold_cv_stats["overall"] = overall_kfold_stats

    logger.info(f"completed {num_folds:d}-fold cross validation")

    return kfold_cv_stats, kfold_split_indices


def _get_compute_description(backend: Backend) -> dict:
    """Returns the compute description for the backend."""
    compute_description = {"num_nodes": backend.num_nodes}

    if torch.cuda.is_available():
        # Assumption: All nodes are of the same instance type.
        # TODO: fix for Ray where workers may be of different skus
        compute_description.update(
            {
                "gpus_per_node": torch.cuda.device_count(),
                "arch_list": torch.cuda.get_arch_list(),
                "gencode_flags": torch.cuda.get_gencode_flags(),
                "devices": {},
            }
        )
        for i in range(torch.cuda.device_count()):
            compute_description["devices"][i] = {
                "gpu_type": torch.cuda.get_device_name(i),
                "device_capability": torch.cuda.get_device_capability(i),
                "device_properties": str(torch.cuda.get_device_properties(i)),
            }

    return compute_description


@PublicAPI
def get_experiment_description(
    config: ModelConfigDict,
    dataset: str | dict | pd.DataFrame | None = None,
    training_set: str | dict | pd.DataFrame | None = None,
    validation_set: str | dict | pd.DataFrame | None = None,
    test_set: str | dict | pd.DataFrame | None = None,
    training_set_metadata: TrainingSetMetadataDict | None = None,
    data_format: str | None = None,
    backend: Backend | None = None,
    random_seed: int | None = None,
) -> dict:
    description = OrderedDict()
    description["ludwig_version"] = LUDWIG_VERSION
    description["command"] = " ".join(sys.argv)

    commit_hash = get_commit_hash()
    if commit_hash is not None:
        description["commit_hash"] = commit_hash[:12]

    if random_seed is not None:
        description["random_seed"] = random_seed

    if isinstance(dataset, str):
        description["dataset"] = dataset
    if isinstance(training_set, str):
        description["training_set"] = training_set
    if isinstance(validation_set, str):
        description["validation_set"] = validation_set
    if isinstance(test_set, str):
        description["test_set"] = test_set
    if training_set_metadata is not None:
        description["training_set_metadata"] = training_set_metadata

    # determine data format if not provided or auto
    if not data_format or data_format == "auto":
        data_format = figure_data_format(dataset, training_set, validation_set, test_set)

    if data_format:
        description["data_format"] = str(data_format)

    description["config"] = config
    description["torch_version"] = torch.__version__
    description["compute"] = _get_compute_description(backend)

    return description
