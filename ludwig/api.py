# !/usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
import traceback
from collections import OrderedDict
from pprint import pformat
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from marshmallow_dataclass import dataclass
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
from ludwig.globals import (
    LUDWIG_VERSION,
    MODEL_HYPERPARAMETERS_FILE_NAME,
    set_disable_progressbar,
    TRAIN_SET_METADATA_FILE_NAME,
)
from ludwig.models.base import BaseModel
from ludwig.models.calibrator import Calibrator
from ludwig.models.inference import InferenceModule, save_ludwig_model_for_inference
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
from ludwig.utils.misc_utils import (
    get_commit_hash,
    get_file_names,
    get_from_registry,
    get_output_directory,
    set_saved_weights_in_checkpoint_flag,
)
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.torch_utils import DEVICE
from ludwig.utils.trainer_utils import get_training_report
from ludwig.utils.types import DataFrame, TorchDevice

logger = logging.getLogger(__name__)


@PublicAPI
@dataclass
class EvaluationFrequency:  # noqa F821
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
class TrainingStats:  # noqa F821
    """Training stats were previously represented as a tuple or a dict.

    This class replaces those while preserving dict and tuple-like behavior (unpacking, [] access).
    """

    training: Dict[str, Any]
    validation: Dict[str, Any]
    test: Dict[str, Any]
    evaluation_frequency: EvaluationFrequency = dataclasses.field(default_factory=EvaluationFrequency)

    # TODO(daniel): deprecate multiple return value unpacking and dictionary-style element access
    def __iter__(self):
        return iter((self.training, self.test, self.validation))

    def __contains__(self, key):
        return (
            (key == TRAINING and self.training)
            or (key == VALIDATION and self.validation)
            or (key == TEST and self.test)
        )

    def __getitem__(self, key):
        # Supports dict-style [] element access for compatibility.
        return {TRAINING: self.training, VALIDATION: self.validation, TEST: self.test}[key]


@PublicAPI
@dataclass
class PreprocessedDataset:  # noqa F821
    training_set: Dataset
    validation_set: Dataset
    test_set: Dataset
    training_set_metadata: TrainingSetMetadataDict

    # TODO(daniel): deprecate multiple return value unpacking and indexed access
    def __iter__(self):
        return iter((self.training_set, self.validation_set, self.test_set, self.training_set_metadata))

    def __getitem__(self, index):
        return (self.training_set, self.validation_set, self.test_set, self.training_set_metadata)[index]


@PublicAPI
@dataclass
class TrainingResults:  # noqa F821
    train_stats: TrainingStats
    preprocessed_data: PreprocessedDataset
    output_directory: str

    def __iter__(self):
        """Supports tuple-style return value unpacking ex.

        train_stats, training_set, output_dir = model.train(...)
        """
        return iter((self.train_stats, self.preprocessed_data, self.output_directory))

    def __getitem__(self, index):
        """Provides indexed getter ex.

        train_stats = model.train(...)[0]
        """
        return (self.train_stats, self.preprocessed_data, self.output_directory)[index]


@PublicAPI
class LudwigModel:
    """Class that allows access to high level Ludwig functionalities.

    # Inputs

    :param config: (Union[str, dict]) in-memory representation of
            config or string path to a YAML config file.
    :param logging_level: (int) Log level that will be sent to stderr.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param gpus: (Union[str, int, List[int]], default: `None`) GPUs
        to use (it uses the same syntax of CUDA_VISIBLE_DEVICES)
    :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
        [0, 1] allowed to allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow Torch
        to use multithreading parallelism to improve performance at the
        cost of determinism.

    # Example usage:

    ```python
    from ludwig.api import LudwigModel
    ```

    Train a model:

    ```python
    config = {...}
    ludwig_model = LudwigModel(config)
    train_stats, _, _ = ludwig_model.train(dataset=file_path)
    ```

    or

    ```python
    train_stats, _, _ = ludwig_model.train(dataset=dataframe)
    ```

    If you have already trained a model you can load it and use it to predict

    ```python
    ludwig_model = LudwigModel.load(model_dir)
    ```

    Predict:

    ```python
    predictions, _ = ludwig_model.predict(dataset=file_path)
    ```

    or

    ```python
    predictions, _ = ludwig_model.predict(dataset=dataframe)
    ```

    Evaluation:

    ```python
    eval_stats, _, _ = ludwig_model.evaluate(dataset=file_path)
    ```

    or

    ```python
    eval_stats, _, _ = ludwig_model.evaluate(dataset=dataframe)
    ```
    """

    def __init__(
        self,
        config: Union[str, dict],
        logging_level: int = logging.ERROR,
        backend: Optional[Union[Backend, str]] = None,
        gpus: Optional[Union[str, int, List[int]]] = None,
        gpu_memory_limit: Optional[float] = None,
        allow_parallel_threads: bool = True,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Constructor for the Ludwig Model class.

        # Inputs

        :param config: (Union[str, dict]) in-memory representation of
            config or string path to a YAML config file.
        :param logging_level: (int) Log level that will be sent to stderr.
        :param backend: (Union[Backend, str]) `Backend` or string name
            of backend to use to execute preprocessing / training steps.
        :param gpus: (Union[str, int, List[int]], default: `None`) GPUs
            to use (it uses the same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
            [0, 1] allowed to allocate per GPU device.
        :param allow_parallel_threads: (bool, default: `True`) allow Torch
            to use multithreading parallelism to improve performance at the
            cost of determinism.
        :param callbacks: (list, default: `None`) a list of
              `ludwig.callbacks.Callback` objects that provide hooks into the
               Ludwig pipeline.

        # Return

        :return: (None) `None`
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
        self.callbacks = callbacks if callbacks is not None else []

        # setup PyTorch env (GPU allocation, etc.)
        self.backend.initialize_pytorch(
            gpus=gpus, gpu_memory_limit=gpu_memory_limit, allow_parallel_threads=allow_parallel_threads
        )

        # setup model
        self.model = None
        self.training_set_metadata: Optional[str, dict] = None

        # online training state
        self._online_trainer = None

    def train(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        training_set: Optional[Union[str, dict, pd.DataFrame, Dataset]] = None,
        validation_set: Optional[Union[str, dict, pd.DataFrame, Dataset]] = None,
        test_set: Optional[Union[str, dict, pd.DataFrame, Dataset]] = None,
        training_set_metadata: Optional[Union[str, dict]] = None,
        data_format: Optional[str] = None,
        experiment_name: str = "api_experiment",
        model_name: str = "run",
        model_resume_path: Optional[str] = None,
        skip_save_training_description: bool = False,
        skip_save_training_statistics: bool = False,
        skip_save_model: bool = False,
        skip_save_progress: bool = False,
        skip_save_log: bool = False,
        skip_save_processed_input: bool = False,
        output_directory: Optional[str] = "results",
        random_seed: int = default_random_seed,
        **kwargs,
    ) -> TrainingResults:
        """This function is used to perform a full training of the model on the specified dataset.

        During training if the skip parameters are False
        the model and statistics will be saved in a directory
        `[output_dir]/[experiment_name]_[model_name]_n` where all variables are
        resolved to user specified ones and `n` is an increasing number
        starting from 0 used to differentiate among repeated runs.

        # Inputs

        :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing the entire dataset to be used in the experiment.
            If it has a split column, it will be used for splitting
            (0 for train, 1 for validation, 2 for test),
            otherwise the dataset will be randomly split.
        :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing training data.
        :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing validation data.
        :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing test data.
        :param training_set_metadata: (Union[str, dict], default: `None`)
            metadata JSON file or loaded metadata. Intermediate preprocessed
            structure containing the mappings of the input dataset created the
            first time an input file is used in the same directory with the
            same name and a '.meta.json' extension.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`,
            `'feather'`, `'fwf'`,
            `'hdf5'` (cache file produced during previous training),
            `'html'` (file containing a single HTML `<table>`),
            `'json'`, `'jsonl'`, `'parquet'`,
            `'pickle'` (pickled Pandas DataFrame),
            `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
        :param experiment_name: (str, default: `'experiment'`) name for
            the experiment.
        :param model_name: (str, default: `'run'`) name of the model that is
            being used.
        :param model_resume_path: (str, default: `None`) resumes training of
            the model from the path specified. The config is restored.
            In addition to config, training statistics, loss for each
            epoch and the state of the optimizer are restored such that
            training can be effectively continued from a previously interrupted
            training process.
        :param skip_save_training_description: (bool, default: `False`)
            disables saving the description JSON file.
        :param skip_save_training_statistics: (bool, default: `False`)
            disables saving training statistics JSON file.
        :param skip_save_model: (bool, default: `False`) disables
            saving model weights and hyperparameters each time the model
            improves. By default Ludwig saves model weights after each epoch
            the validation metric improves, but if the model is really big
            that can be time consuming. If you do not want to keep
            the weights and just find out what performance a model can get
            with a set of hyperparameters, use this parameter to skip it,
            but the model will not be loadable later on and the returned model
            will have the weights obtained at the end of training, instead of
            the weights of the epoch with the best validation performance.
        :param skip_save_progress: (bool, default: `False`) disables saving
            progress each epoch. By default Ludwig saves weights and stats
            after each epoch for enabling resuming of training, but if
            the model is really big that can be time consuming and will uses
            twice as much space, use this parameter to skip it, but training
            cannot be resumed later on.
        :param skip_save_log: (bool, default: `False`) disables saving
            TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
            but if it is not needed turning it off can slightly increase the
            overall speed.
        :param skip_save_processed_input: (bool, default: `False`) if input
            dataset is provided it is preprocessed and cached by saving an HDF5
            and JSON files to avoid running the preprocessing again. If this
            parameter is `False`, the HDF5 and JSON file are not saved.
        :param output_directory: (str, default: `'results'`) the directory that
            will contain the training statistics, TensorBoard logs, the saved
            model and the training progress files.
        :param random_seed: (int, default: `42`) a random seed that will be
            used anywhere there is a call to a random number generator: data
            splitting, parameter initialization and training set shuffling
        :param kwargs: (dict, default: {}) a dictionary of optional parameters.

        # Return

        :return: (Tuple[Dict, Union[Dict, pd.DataFrame], str]) tuple containing
            `(training_statistics, preprocessed_data, output_directory)`.
            `training_statistics` is a nested dictionary of dataset -> feature_name -> metric_name -> List of metrics.
                Each metric corresponds to each training checkpoint.
            `preprocessed_data` is the tuple containing these three data sets
            `(training_set, validation_set, test_set)`.
            `output_directory` filepath to where training results are stored.
        """
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
                preprocessed_data = (training_set, validation_set, test_set, training_set_metadata)
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
                (training_set, validation_set, test_set, training_set_metadata) = preprocessed_data

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
                self._tune_batch_size(trainer, training_set, random_seed=random_seed)

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
                    (self.model, train_trainset_stats, train_valiset_stats, train_testset_stats) = train_stats

                    # Calibrates output feature probabilities on validation set if calibration is enabled.
                    # Must be done after training, and before final model parameters are saved.
                    if self.backend.is_coordinator():
                        calibrator = Calibrator(
                            self.model,
                            self.backend,
                            batch_size=trainer.eval_batch_size,
                        )
                        if calibrator.calibration_enabled():
                            if validation_set is None:
                                logger.warning(
                                    "Calibration uses validation set, but no validation split specified."
                                    "Will use training set for calibration."
                                    "Recommend providing a validation set when using calibration."
                                )
                                calibrator.train_calibration(training_set, TRAINING)
                            elif len(validation_set) < MIN_DATASET_SPLIT_ROWS:
                                logger.warning(
                                    f"Validation set size ({len(validation_set)} rows) is too small for calibration."
                                    "Will use training set for calibration."
                                    f"Validation set much have at least {MIN_DATASET_SPLIT_ROWS} rows."
                                )
                                calibrator.train_calibration(training_set, TRAINING)
                            else:
                                calibrator.train_calibration(validation_set, VALIDATION)
                        if not skip_save_model:
                            self.model.save(model_dir)

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

                # Ensure model weights are saved to the driver if training was done remotely
                if self.backend.is_coordinator() and not skip_save_model:
                    self.model.save(model_dir)

                if self.is_merge_and_unload_set():
                    # For an LLM model trained with a LoRA adapter, handle merge and unload postprocessing directives.
                    self.model.merge_and_unload(progressbar=self.config_obj.adapter.postprocessor.progressbar)

                    # Also: Ensure that the full model weights are saved to the driver if training was done remotely.
                    if self.backend.is_coordinator() and not skip_save_model:
                        self.model.save_base_model(model_dir)

                # Synchronize model weights between workers
                self.backend.sync_model(self.model)

                print_boxed("FINISHED")
                return TrainingResults(train_stats, preprocessed_data, output_url)

    def train_online(
        self,
        dataset: Union[str, dict, pd.DataFrame],
        training_set_metadata: Optional[Union[str, dict]] = None,
        data_format: str = "auto",
        random_seed: int = default_random_seed,
    ) -> None:
        """Performs one epoch of training of the model on `dataset`.

        # Inputs

        :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing the entire dataset to be used in the experiment.
            If it has a split column, it will be used for splitting (0 for train,
            1 for validation, 2 for test), otherwise the dataset will be
            randomly split.
        :param training_set_metadata: (Union[str, dict], default: `None`)
            metadata JSON file or loaded metadata.  Intermediate preprocessed
        structure containing the mappings of the input
            dataset created the first time an input file is used in the same
            directory with the same name and a '.meta.json' extension.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`, `'hdf5'` (cache file produced during previous training),
            `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
            `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
            `'stata'`, `'tsv'`.
        :param random_seed: (int, default: `42`) a random seed that is going to be
               used anywhere there is a call to a random number generator: data
               splitting, parameter initialization and training set shuffling

        # Return

        :return: (None) `None`
        """
        training_set_metadata = training_set_metadata or self.training_set_metadata
        preprocessing_params = get_preprocessing_params(self.config_obj)

        with provision_preprocessing_workers(self.backend):
            # TODO (Connor): Refactor to use self.config_obj
            training_dataset, _, _, training_set_metadata = preprocess_for_training(
                self.config_obj.to_dict(),
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

            self._tune_batch_size(self._online_trainer, dataset, random_seed=random_seed)

        self.model = self._online_trainer.train_online(training_dataset)

    def _tune_batch_size(self, trainer, dataset, random_seed: int = default_random_seed):
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
            # Models like GBMs don't have batch sizes to be tuned
            return

        # Render the batch size and gradient accumulation steps prior to batch size tuning. This is needed in the event
        # the effective_batch_size and gradient_accumulation_steps are set explicitly, but batch_size is AUTO. In this
        # case, we can infer the batch_size directly without tuning.
        num_workers = self.backend.num_training_workers
        self.config_obj.trainer.update_batch_size_grad_accum(num_workers)

        # TODO (ASN): add support for substitute_with_max parameter
        # TODO(travis): detect train and eval batch sizes separately (enable / disable gradients)
        if self.config_obj.trainer.batch_size == AUTO:
            if self.backend.supports_batch_size_tuning():
                tuned_batch_size = trainer.tune_batch_size(
                    self.config_obj.to_dict(), dataset, random_seed=random_seed, tune_for_training=True
                )
            else:
                logger.warning(
                    f"Backend {self.backend.BACKEND_TYPE} does not support batch size tuning, "
                    f"using fallback training batch size {FALLBACK_BATCH_SIZE}."
                )
                tuned_batch_size = FALLBACK_BATCH_SIZE

            # TODO(travis): pass these in as args to trainer when we call train,
            #  to avoid setting state on possibly remote trainer
            self.config_obj.trainer.batch_size = tuned_batch_size

            # Re-render the gradient_accumulation_steps to account for the explicit batch size.
            self.config_obj.trainer.update_batch_size_grad_accum(num_workers)

        if self.config_obj.trainer.eval_batch_size in {AUTO, None}:
            if self.backend.supports_batch_size_tuning():
                tuned_batch_size = trainer.tune_batch_size(
                    self.config_obj.to_dict(), dataset, random_seed=random_seed, tune_for_training=False
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

    def predict(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        data_format: str = None,
        split: str = FULL,
        batch_size: int = 128,
        generation_config: Optional[dict] = None,
        skip_save_unprocessed_output: bool = True,
        skip_save_predictions: bool = True,
        output_directory: str = "results",
        return_type: Union[str, dict, pd.DataFrame] = pd.DataFrame,
        callbacks: Optional[List[Callback]] = None,
        **kwargs,
    ) -> Tuple[Union[dict, pd.DataFrame], str]:
        """Using a trained model, make predictions from the provided dataset.

        # Inputs

        :param dataset: (Union[str, dict, pandas.DataFrame]): source containing the entire dataset to be evaluated.
        :param data_format: (str, default: `None`) format to interpret data sources. Will be inferred automatically
            if not specified.  Valid formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`, `'hdf5'` (cache file produced during previous training), `'html'` (file containing a single
            HTML `<table>`), `'json'`, `'jsonl'`, `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`,
            `'spss'`, `'stata'`, `'tsv'`.
        :param split: (str, default= `'full'`):  if the input dataset contains a split column, this parameter
            indicates which split of the data to use. Possible values are `'full'`, `'training'`, `'validation'`,
            `'test'`.
        :param batch_size: (int, default: 128) size of batch to use when making predictions.
        :param generation_config: (Dict, default: `None`) config for the generation of the
            predictions. If `None`, the config that was used during model training is
            used. This is only used if the model type is LLM. Otherwise, this parameter is
            ignored. See
            [Large Language Models](https://ludwig.ai/latest/configuration/large_language_model/#generation) under
            "Generation" for an example generation config.
        :param skip_save_unprocessed_output: (bool, default: `True`) if this parameter is `False`, predictions and
            their probabilities are saved in both raw unprocessed numpy files containing tensors and as
            postprocessed CSV files (one for each output feature). If this parameter is `True`, only the CSV ones
            are saved and the numpy ones are skipped.
        :param skip_save_predictions: (bool, default: `True`) skips saving test predictions CSV files.
        :param output_directory: (str, default: `'results'`) the directory that will contain the training
            statistics, TensorBoard logs, the saved model and the training progress files.
        :param return_type: (Union[str, dict, pandas.DataFrame], default: pd.DataFrame) indicates the format of the
            returned predictions.
        :param callbacks: (Optional[List[Callback]], default: None) optional list of callbacks to use during this
            predict operation. Any callbacks already registered to the model will be preserved.

        # Return

        :return `(predictions, output_directory)`: (Tuple[Union[dict, pd.DataFrame], str])
            `predictions` predictions from the provided dataset,
            `output_directory` filepath string to where data was stored.
        """
        self._check_initialization()

        # preprocessing
        logger.debug("Preprocessing")
        dataset, _ = preprocess_for_prediction(  # TODO (Connor): Refactor to use self.config_obj
            self.config_obj.to_dict(),
            dataset=dataset,
            training_set_metadata=self.training_set_metadata,
            data_format=data_format,
            split=split,
            include_outputs=False,
            backend=self.backend,
            callbacks=self.callbacks + (callbacks or []),
        )

        logger.debug("Predicting")
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

            logger.debug("Postprocessing")
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

            return converted_postproc_predictions, output_directory

    def evaluate(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        data_format: Optional[str] = None,
        split: str = FULL,
        batch_size: Optional[int] = None,
        skip_save_unprocessed_output: bool = True,
        skip_save_predictions: bool = True,
        skip_save_eval_stats: bool = True,
        collect_predictions: bool = False,
        collect_overall_stats: bool = False,
        output_directory: str = "results",
        return_type: Union[str, dict, pd.DataFrame] = pd.DataFrame,
        **kwargs,
    ) -> Tuple[dict, Union[dict, pd.DataFrame], str]:
        """This function is used to predict the output variables given the input variables using the trained model
        and compute test statistics like performance measures, confusion matrices and the like.

        # Inputs
        :param dataset: (Union[str, dict, pandas.DataFrame]) source containing
            the entire dataset to be evaluated.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`, `'hdf5'` (cache file produced during previous training),
            `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
            `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
            `'stata'`, `'tsv'`.
        :param split: (str, default=`'full'`): if the input dataset contains
            a split column, this parameter indicates which split of the data
            to use. Possible values are `'full'`, `'training'`, `'validation'`, `'test'`.
        :param batch_size: (int, default: None) size of batch to use when making
            predictions. Defaults to model config eval_batch_size
        :param skip_save_unprocessed_output: (bool, default: `True`) if this
            parameter is `False`, predictions and their probabilities are saved
            in both raw unprocessed numpy files containing tensors and as
            postprocessed CSV files (one for each output feature).
            If this parameter is `True`, only the CSV ones are saved and the
            numpy ones are skipped.
        :param skip_save_predictions: (bool, default: `True`) skips saving
            test predictions CSV files.
        :param skip_save_eval_stats: (bool, default: `True`) skips saving
            test statistics JSON file.
        :param collect_predictions: (bool, default: `False`) if `True`
            collects post-processed predictions during eval.
        :param collect_overall_stats: (bool, default: False) if `True`
            collects overall stats during eval.
        :param output_directory: (str, default: `'results'`) the directory that
            will contain the training statistics, TensorBoard logs, the saved
            model and the training progress files.
        :param return_type: (Union[str, dict, pd.DataFrame], default: pandas.DataFrame) indicates
            the format to of the returned predictions.

        # Return
        :return: (`evaluation_statistics`, `predictions`, `output_directory`)
            `evaluation_statistics` dictionary containing evaluation performance
                statistics,
            `postprocess_predictions` contains predicted values,
            `output_directory` is location where results are stored.
        """
        self._check_initialization()

        for callback in self.callbacks:
            callback.on_evaluation_start()

        # preprocessing
        logger.debug("Preprocessing")
        dataset, training_set_metadata = preprocess_for_prediction(  # TODO (Connor): Refactor to use self.config_obj
            self.config_obj.to_dict(),
            dataset=dataset,
            training_set_metadata=self.training_set_metadata,
            data_format=data_format,
            split=split,
            include_outputs=True,
            backend=self.backend,
            callbacks=self.callbacks,
        )

        # Fallback to use eval_batch_size or batch_size if not provided
        if batch_size is None:
            # Requires dictionary getter since gbm config does not have a batch_size param
            batch_size = self.config_obj.trainer.to_dict().get(
                EVAL_BATCH_SIZE, None
            ) or self.config_obj.trainer.to_dict().get(BATCH_SIZE, None)

        logger.debug("Predicting")
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
                    of_name: {**eval_stats[of_name], **overall_stats[of_name]}
                    # account for presence of 'combined' key
                    if of_name in overall_stats else {**eval_stats[of_name]}
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
                logger.debug("Postprocessing")
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
        data_format: Optional[str] = None,
        horizon: int = 1,
        output_directory: Optional[str] = None,
        output_format: str = "parquet",
    ) -> DataFrame:
        # TODO(travis): WIP
        dataset, _, _, _ = load_dataset_uris(dataset, None, None, None, self.backend)
        if isinstance(dataset, CacheableDataset):
            dataset = dataset.unwrap()
        dataset = load_dataset(dataset, data_format=data_format, df_lib=self.backend.df_engine.df_lib)

        window_sizes = [
            feature.preprocessing.window_size
            for feature in self.config_obj.input_features
            if feature.type == TIMESERIES
        ]
        if not window_sizes:
            raise ValueError("Forecasting requires at least one input feature of type `timeseries`.")

        # TODO(travis): there's a lot of redundancy in this approach, since we are preprocessing the same DataFrame
        # multiple times with only a small number of features (the horizon) being appended each time.
        # A much better approach would be to only preprocess a single row, but incorporating the row-level embedding
        # over the window_size of rows precending it, then performing the model forward pass on only that row of
        # data.
        max_lookback_window_size = max(window_sizes)
        total_forecasted = 0
        while total_forecasted < horizon:
            # We only need the last `window_size` worth of rows to forecast the next value
            dataset = dataset.tail(max_lookback_window_size)

            # Run through preprocessing and prediction to obtain row-wise next values
            # TODO(travis): can optimize the preprocessing part here, since we only need to preprocess / predict
            # the last row, not the last `window_size` rows.
            preds, _ = self.predict(dataset, skip_save_predictions=True, skip_save_unprocessed_output=True)

            next_series = {}
            for feature in self.config_obj.output_features:
                if feature.type == TIMESERIES:
                    key = f"{feature.name}_predictions"
                    next_series[feature.column] = pd.Series(preds[key].iloc[-1])

            next_preds = pd.DataFrame(next_series)
            dataset = pd.concat([dataset, next_preds], axis=0).reset_index(drop=True)
            total_forecasted += len(next_preds)

        horizon_df = dataset.tail(total_forecasted).head(horizon)
        return_cols = [feature.column for feature in self.config_obj.output_features if feature.type == TIMESERIES]
        results_df = horizon_df[return_cols]

        if output_directory is not None:
            if self.backend.is_coordinator():
                # TODO(travis): generalize this to support any pandas output format
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
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        training_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        validation_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        test_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        training_set_metadata: Optional[Union[str, dict]] = None,
        data_format: Optional[str] = None,
        experiment_name: str = "experiment",
        model_name: str = "run",
        model_resume_path: Optional[str] = None,
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
    ) -> Tuple[Optional[dict], TrainingStats, PreprocessedDataset, str]:
        """Trains a model on a dataset's training and validation splits and uses it to predict on the test split.
        It saves the trained model and the statistics of training and testing.

        # Inputs
        :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing the entire dataset to be used in the experiment.
            If it has a split column, it will be used for splitting (0 for train,
            1 for validation, 2 for test), otherwise the dataset will be
            randomly split.
        :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing training data.
        :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing validation data.
        :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
            source containing test data.
        :param training_set_metadata: (Union[str, dict], default: `None`)
            metadata JSON file or loaded metadata.  Intermediate preprocessed
        structure containing the mappings of the input
            dataset created the first time an input file is used in the same
            directory with the same name and a '.meta.json' extension.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`, `'hdf5'` (cache file produced during previous training),
            `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
            `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
            `'stata'`, `'tsv'`.
        :param experiment_name: (str, default: `'experiment'`) name for
            the experiment.
        :param model_name: (str, default: `'run'`) name of the model that is
            being used.
        :param model_resume_path: (str, default: `None`) resumes training of
            the model from the path specified. The config is restored.
            In addition to config, training statistics and loss for
            epoch and the state of the optimizer are restored such that
            training can be effectively continued from a previously interrupted
            training process.
        :param eval_split: (str, default: `test`) split on which
            to perform evaluation. Valid values are `training`, `validation`
            and `test`.
        :param skip_save_training_description: (bool, default: `False`) disables
            saving the description JSON file.
        :param skip_save_training_statistics: (bool, default: `False`) disables
            saving training statistics JSON file.
        :param skip_save_model: (bool, default: `False`) disables
            saving model weights and hyperparameters each time the model
            improves. By default Ludwig saves model weights after each epoch
            the validation metric improves, but if the model is really big
            that can be time consuming. If you do not want to keep
            the weights and just find out what performance a model can get
            with a set of hyperparameters, use this parameter to skip it,
            but the model will not be loadable later on and the returned model
            will have the weights obtained at the end of training, instead of
            the weights of the epoch with the best validation performance.
        :param skip_save_progress: (bool, default: `False`) disables saving
            progress each epoch. By default Ludwig saves weights and stats
            after each epoch for enabling resuming of training, but if
            the model is really big that can be time consuming and will uses
            twice as much space, use this parameter to skip it, but training
            cannot be resumed later on.
        :param skip_save_log: (bool, default: `False`) disables saving
            TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
            but if it is not needed turning it off can slightly increase the
            overall speed.
        :param skip_save_processed_input: (bool, default: `False`) if input
            dataset is provided it is preprocessed and cached by saving an HDF5
            and JSON files to avoid running the preprocessing again. If this
            parameter is `False`, the HDF5 and JSON file are not saved.
        :param skip_save_unprocessed_output: (bool, default: `False`) by default
            predictions and their probabilities are saved in both raw
            unprocessed numpy files containing tensors and as postprocessed
            CSV files (one for each output feature). If this parameter is True,
            only the CSV ones are saved and the numpy ones are skipped.
        :param skip_save_predictions: (bool, default: `False`) skips saving test
            predictions CSV files
        :param skip_save_eval_stats: (bool, default: `False`) skips saving test
            statistics JSON file
        :param skip_collect_predictions: (bool, default: `False`) skips
            collecting post-processed predictions during eval.
        :param skip_collect_overall_stats: (bool, default: `False`) skips
            collecting overall stats during eval.
        :param output_directory: (str, default: `'results'`) the directory that
            will contain the training statistics, TensorBoard logs, the saved
            model and the training progress files.
        :param random_seed: (int: default: 42) random seed used for weights
            initialization, splits and any other random function.

        # Return
        :return: (Tuple[dict, dict, tuple, str))
            `(evaluation_statistics, training_statistics, preprocessed_data, output_directory)`
            `evaluation_statistics` dictionary with evaluation performance
                statistics on the test_set,
            `training_statistics` is a nested dictionary of dataset -> feature_name -> metric_name -> List of metrics.
                Each metric corresponds to each training checkpoint.
            `preprocessed_data` tuple containing preprocessed
            `(training_set, validation_set, test_set)`, `output_directory`
            filepath string to where results are stored.
        """
        if self._user_config.get(HYPEROPT):
            print_boxed("WARNING")
            logger.warning(HYPEROPT_WARNING)

        (train_stats, preprocessed_data, output_directory) = self.train(
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

        (training_set, validation_set, test_set, training_set_metadata) = preprocessed_data

        eval_set = validation_set
        if eval_split == TRAINING:
            eval_set = training_set
        elif eval_split == VALIDATION:
            eval_set = validation_set
        elif eval_split == TEST:
            eval_set = test_set
        else:
            logger.warning(f"Eval split {eval_split} not supported. " f"Using validation set instead")

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
            logger.warning(f"The evaluation set {eval_set} was not provided. " f"Skipping evaluation")
            eval_stats = None

        return eval_stats, train_stats, preprocessed_data, output_directory

    def collect_weights(self, tensor_names: List[str] = None, **kwargs) -> list:
        """Load a pre-trained model and collect the tensors with a specific name.

        # Inputs
        :param tensor_names: (list, default: `None`) List of tensor names to collect
            weights

        # Return
        :return: (list) List of tensors
        """
        self._check_initialization()
        collected_tensors = self.model.collect_weights(tensor_names)
        return collected_tensors

    def collect_activations(
        self,
        layer_names: List[str],
        dataset: Union[str, Dict[str, list], pd.DataFrame],
        data_format: Optional[str] = None,
        split: str = FULL,
        batch_size: int = 128,
        **kwargs,
    ) -> list:
        """Loads a pre-trained model model and input data to collect the values of the activations contained in the
        tensors.

        # Inputs
        :param layer_names: (list) list of strings for layer names in the model
            to collect activations.
        :param dataset: (Union[str, Dict[str, list], pandas.DataFrame]) source
            containing the data to make predictions.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`, `'hdf5'` (cache file produced during previous training),
            `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
            `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
            `'stata'`, `'tsv'`.
        :param split: (str, default= `'full'`): if the input dataset contains
            a split column, this parameter indicates which split of the data
            to use. Possible values are `'full'`, `'training'`, `'validation'`, `'test'`.
        :param batch_size: (int, default: 128) size of batch to use when making
            predictions.

        # Return
        :return: (list) list of collected tensors.
        """
        self._check_initialization()

        # preprocessing
        logger.debug("Preprocessing")
        dataset, training_set_metadata = preprocess_for_prediction(  # TODO (Connor): Refactor to use self.config_obj
            self.config_obj.to_dict(),
            dataset=dataset,
            training_set_metadata=self.training_set_metadata,
            data_format=data_format,
            split=split,
            include_outputs=False,
        )

        logger.debug("Predicting")
        with self.backend.create_predictor(self.model, batch_size=batch_size) as predictor:
            activations = predictor.batch_collect_activations(
                layer_names,
                dataset,
            )

            return activations

    def preprocess(
        self,
        dataset: Optional[Union[str, dict, pd.DataFrame]] = None,
        training_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        validation_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        test_set: Optional[Union[str, dict, pd.DataFrame]] = None,
        training_set_metadata: Optional[Union[str, dict]] = None,
        data_format: Optional[str] = None,
        skip_save_processed_input: bool = True,
        random_seed: int = default_random_seed,
        **kwargs,
    ) -> PreprocessedDataset:
        """This function is used to preprocess data.

        # Args:
            :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
                source containing the entire dataset to be used in the experiment.
                If it has a split column, it will be used for splitting
                (0 for train, 1 for validation, 2 for test),
                otherwise the dataset will be randomly split.
            :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
                source containing training data.
            :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
                source containing validation data.
            :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
                source containing test data.
            :param training_set_metadata: (Union[str, dict], default: `None`)
                metadata JSON file or loaded metadata. Intermediate preprocessed
            structure containing the mappings of the input
                dataset created the first time an input file is used in the same
                directory with the same name and a '.meta.json' extension.
            :param data_format: (str, default: `None`) format to interpret data
                sources. Will be inferred automatically if not specified.  Valid
                formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`,
                `'feather'`, `'fwf'`,
                `'hdf5'` (cache file produced during previous training),
                `'html'` (file containing a single HTML `<table>`),
                `'json'`, `'jsonl'`, `'parquet'`,
                `'pickle'` (pickled Pandas DataFrame),
                `'sas'`, `'spss'`, `'stata'`, `'tsv'`.
            :param skip_save_processed_input: (bool, default: `False`) if input
                dataset is provided it is preprocessed and cached by saving an HDF5
                and JSON files to avoid running the preprocessing again. If this
                parameter is `False`, the HDF5 and JSON file are not saved.
            :param random_seed: (int, default: `42`) a random seed that will be
                used anywhere there is a call to a random number generator: data
                splitting, parameter initialization and training set shuffling

        # Returns:
            :return: (PreprocessedDataset) data structure containing
                `(proc_training_set, proc_validation_set, proc_test_set, training_set_metadata)`.

        # Raises:
            RuntimeError: An error occured while preprocessing the data. Examples include training dataset
                being empty after preprocessing, lazy loading not being supported with RayBackend, etc.
        """
        print_boxed("PREPROCESSING")

        for callback in self.callbacks:
            callback.on_preprocess_start(self.config_obj.to_dict())

        preprocessing_params = get_preprocessing_params(self.config_obj)

        proc_training_set = proc_validation_set = proc_test_set = None
        try:
            with provision_preprocessing_workers(self.backend):
                # TODO (Connor): Refactor to use self.config_obj
                preprocessed_data = preprocess_for_training(
                    self.config_obj.to_dict(),
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

            (proc_training_set, proc_validation_set, proc_test_set, training_set_metadata) = preprocessed_data

            return PreprocessedDataset(proc_training_set, proc_validation_set, proc_test_set, training_set_metadata)
        except Exception as e:
            raise RuntimeError(f"Caught exception during model preprocessing: {str(e)}") from e
        finally:
            for callback in self.callbacks:
                callback.on_preprocess_end(proc_training_set, proc_validation_set, proc_test_set, training_set_metadata)

    @staticmethod
    def load(
        model_dir: str,
        logging_level: int = logging.ERROR,
        backend: Optional[Union[Backend, str]] = None,
        gpus: Optional[Union[str, int, List[int]]] = None,
        gpu_memory_limit: Optional[float] = None,
        allow_parallel_threads: bool = True,
        callbacks: List[Callback] = None,
    ) -> "LudwigModel":  # return is an instance of ludwig.api.LudwigModel class
        """This function allows for loading pretrained models.

        # Inputs

        :param model_dir: (str) path to the directory containing the model.
               If the model was trained by the `train` or `experiment` command,
               the model is in `results_dir/experiment_dir/model`.
        :param logging_level: (int, default: 40) log level that will be sent to
            stderr.
        :param backend: (Union[Backend, str]) `Backend` or string name
            of backend to use to execute preprocessing / training steps.
        :param gpus: (Union[str, int, List[int]], default: `None`) GPUs
            to use (it uses the same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
            [0, 1] allowed to allocate per GPU device.
        :param allow_parallel_threads: (bool, default: `True`) allow Torch
            to use
            multithreading parallelism to improve performance at the cost of
            determinism.
        :param callbacks: (list, default: `None`) a list of
            `ludwig.callbacks.Callback` objects that provide hooks into the
            Ludwig pipeline.

        # Return

        :return: (LudwigModel) a LudwigModel object


        # Example usage

        ```python
        ludwig_model = LudwigModel.load(model_dir)
        ```
        """
        # Initialize Horovod and PyTorch before calling `broadcast()` to prevent initializing
        # Torch with default parameters
        backend_param = backend
        backend = initialize_backend(backend)
        backend.initialize_pytorch(
            gpus=gpus, gpu_memory_limit=gpu_memory_limit, allow_parallel_threads=allow_parallel_threads
        )

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
        ludwig_model.model = LudwigModel.create_model(config_obj)

        # load model weights
        ludwig_model.load_weights(model_dir)

        # The LoRA layers appear to be loaded again (perhaps due to a potential bug); hence, we merge and unload again.
        if ludwig_model.is_merge_and_unload_set():
            # For an LLM model trained with a LoRA adapter, handle merge and unload postprocessing directives.
            ludwig_model.model.merge_and_unload(progressbar=config_obj.adapter.postprocessor.progressbar)

        # load train set metadata
        ludwig_model.training_set_metadata = backend.broadcast_return(
            lambda: load_metadata(os.path.join(model_dir, TRAIN_SET_METADATA_FILE_NAME))
        )

        return ludwig_model

    def load_weights(
        self,
        model_dir: str,
    ) -> None:
        """Loads weights from a pre-trained model.

        # Inputs
        :param model_dir: (str) filepath string to location of a pre-trained
            model

        # Return
        :return: `None`

        # Example usage

        ```python
        ludwig_model.load_weights(model_dir)
        ```
        """
        if self.backend.is_coordinator():
            self.model.load(model_dir)

        self.backend.sync_model(self.model)

    def save(self, save_path: str) -> None:
        """This function allows to save models on disk.

        # Inputs

        :param  save_path: (str) path to the directory where the model is
                going to be saved. Both a JSON file containing the model
                architecture hyperparameters and checkpoints files containing
                model weights will be saved.

        # Return

        :return: (None) `None`

        # Example usage

        ```python
        ludwig_model.save(save_path)
        ```
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
        commit_description: Optional[str] = None,
    ) -> bool:
        """Uploads trained model artifacts to the HuggingFace Hub.

        # Inputs

        :param repo_id (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        :param model_path (`str`):
            The path of the saved model. This is the top level directory where
            the models weights as well as other associated training artifacts
            are saved.
        :param private (`bool`, *optional*, defaults to `False`):
            Whether the model repo should be private.
        :param repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if uploading to a dataset or
            space, `None` or `"model"` if uploading to a model. Default is
            `None`.
        :param commit_message (`str`, *optional*):
            The summary / title / first line of the generated commit. Defaults to:
            `f"Upload {path_in_repo} with huggingface_hub"`
        :param commit_description (`str` *optional*):
            The description of the generated commit

        # Returns

        :return: (bool) True for success, False for failure.
        """
        model_service = get_upload_registry()["hf_hub"]
        hub = model_service()
        hub.login()
        upload_status = hub.upload(
            repo_id=repo_id,
            model_path=model_path,
            repo_type=repo_type,
            private=private,
            commit_message=commit_message,
            commit_description=commit_description,
        )
        return upload_status

    def save_config(self, save_path: str) -> None:
        """Save config to specified location.

        # Inputs

        :param save_path: (str) filepath string to save config as a
            JSON file.

        # Return
        :return: `None`
        """
        os.makedirs(save_path, exist_ok=True)
        model_hyperparameters_path = os.path.join(save_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        save_json(model_hyperparameters_path, self.config_obj.to_dict())

    def to_torchscript(
        self,
        model_only: bool = False,
        device: Optional[TorchDevice] = None,
    ):
        """Converts the trained model to Torchscript.

        # Inputs

        :param  model_only (bool, optional): If True, only the ECD model will be converted to Torchscript. Else,
            preprocessing and postprocessing steps will also be converted to Torchscript.
        :param device (TorchDevice, optional): If None, the model will be converted to Torchscript on the same device to
            ensure maximum model parity.

        # Returns

        :return: A torch.jit.ScriptModule that can be used to predict on a dictionary of inputs.
        """
        if device is None:
            device = DEVICE

        self._check_initialization()
        if model_only:
            return self.model.to_torchscript(device)
        else:
            inference_module = InferenceModule.from_ludwig_model(
                self.model, self.config_obj.to_dict(), self.training_set_metadata, device=device
            )
            return torch.jit.script(inference_module)

    def save_torchscript(
        self,
        save_path: str,
        model_only: bool = False,
        device: Optional[TorchDevice] = None,
    ):
        """Saves the Torchscript model to disk.

        # Inputs

        :param save_path (str): The path to the directory where the model will be saved.
        :param model_only (bool, optional): If True, only the ECD model will be converted to Torchscript. Else, the
            preprocessing and postprocessing steps will also be converted to Torchscript.
        :param device (TorchDevice, optional): If None, the model will be converted to Torchscript on the same device to
            ensure maximum model parity.

        # Return

        :return: `None`
        """
        if device is None:
            device = DEVICE

        save_ludwig_model_for_inference(
            save_path,
            self.model,
            self.config_obj.to_dict(),
            self.training_set_metadata,
            model_only=model_only,
            device=device,
        )

    def _check_initialization(self):
        if self.model is None or self._user_config is None or self.training_set_metadata is None:
            raise ValueError("Model has not been trained or loaded")

    def free_gpu_memory(self):
        """Manually moves the model to CPU to force GPU memory to be freed.

        For more context: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/35
        """
        if torch.cuda.is_available():
            self.model.model.to(torch.device("cpu"))
            torch.cuda.empty_cache()

    @staticmethod
    def create_model(config_obj: Union[ModelConfig, dict], random_seed: int = default_random_seed) -> BaseModel:
        """Instantiates BaseModel object.

        # Inputs
        :param config_obj: (Union[Config, dict]) Ludwig config object
        :param random_seed: (int, default: ludwig default random seed) Random
            seed used for weights initialization,
            splits and any other random function.

        # Return
        :return: (ludwig.models.BaseModel) Instance of the Ludwig model object.
        """
        if isinstance(config_obj, dict):
            config_obj = ModelConfig.from_dict(config_obj)
        model_type = get_from_registry(config_obj.model_type, model_type_registry)
        return model_type(config_obj, random_seed=random_seed)

    @staticmethod
    def set_logging_level(logging_level: int) -> None:
        """Sets level for log messages.

        # Inputs

        :param logging_level: (int) Set/Update the logging level. Use logging
        constants like `logging.DEBUG` , `logging.INFO` and `logging.ERROR`.

        # Return

        :return: `None`
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
        """Check whether the encapsulated model is of type LLM and is configured to merge_and_unload QLoRA weights.

        # Return

            :return (bool): whether merge_and_unload should be done.
        """
        # TODO: In the future, it may be possible to move up the model type check into the BaseModel class.
        return self.config_obj.model_type == MODEL_LLM and self.model.is_merge_and_unload_set()


@PublicAPI
def kfold_cross_validate(
    num_folds: int,
    config: Union[dict, str],
    dataset: str = None,
    data_format: str = None,
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
    gpus: Optional[Union[str, int, List[int]]] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    backend: Optional[Union[Backend, str]] = None,
    logging_level: int = logging.INFO,
    **kwargs,
) -> Tuple[dict, dict]:
    """Performs k-fold cross validation and returns result data structures.

    # Inputs

    :param num_folds: (int) number of folds to create for the cross-validation
    :param config: (Union[dict, str]) model specification
           required to build a model. Parameter may be a dictionary or string
           specifying the file path to a yaml configuration file.  Refer to the
           [User Guide](http://ludwig.ai/user_guide/#model-config)
           for details.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used for k_fold processing.
        :param data_format: (str, default: `None`) format to interpret data
            sources. Will be inferred automatically if not specified.  Valid
            formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
            `'fwf'`,
            `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
            `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
            `'stata'`, `'tsv'`.  Currently `hdf5` format is not supported for
            k_fold cross validation.
    :param skip_save_training_description: (bool, default: `False`) disables
            saving the description JSON file.
    :param skip_save_training_statistics: (bool, default: `False`) disables
            saving training statistics JSON file.
    :param skip_save_model: (bool, default: `False`) disables
        saving model weights and hyperparameters each time the model
        improves. By default Ludwig saves model weights after each epoch
        the validation metric improves, but if the model is really big
        that can be time consuming. If you do not want to keep
        the weights and just find out what performance a model can get
        with a set of hyperparameters, use this parameter to skip it,
        but the model will not be loadable later on and the returned model
        will have the weights obtained at the end of training, instead of
        the weights of the epoch with the best validation performance.
    :param skip_save_progress: (bool, default: `False`) disables saving
           progress each epoch. By default Ludwig saves weights and stats
           after each epoch for enabling resuming of training, but if
           the model is really big that can be time consuming and will uses
           twice as much space, use this parameter to skip it, but training
           cannot be resumed later on.
    :param skip_save_log: (bool, default: `False`) disables saving TensorBoard
           logs. By default Ludwig saves logs for the TensorBoard, but if it
           is not needed turning it off can slightly increase the
           overall speed.
    :param skip_save_processed_input: (bool, default: `False`) if input
        dataset is provided it is preprocessed and cached by saving an HDF5
        and JSON files to avoid running the preprocessing again. If this
        parameter is `False`, the HDF5 and JSON file are not saved.
    :param skip_save_predictions: (bool, default: `False`) skips saving test
            predictions CSV files.
    :param skip_save_eval_stats: (bool, default: `False`) skips saving test
            statistics JSON file.
    :param skip_collect_predictions: (bool, default: `False`) skips collecting
            post-processed predictions during eval.
    :param skip_collect_overall_stats: (bool, default: `False`) skips collecting
            overall stats during eval.
    :param output_directory: (str, default: `'results'`) the directory that
        will contain the training statistics, TensorBoard logs, the saved
        model and the training progress files.
    :param random_seed: (int, default: `42`) Random seed
            used for weights initialization,
           splits and any other random function.
    :param gpus: (list, default: `None`) list of GPUs that are available
            for training.
    :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
            [0, 1] allowed to allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow Torch to
            use multithreading parallelism
           to improve performance at the cost of determinism.
    :param backend: (Union[Backend, str]) `Backend` or string name
            of backend to use to execute preprocessing / training steps.
    :param logging_level: (int, default: INFO) log level to send to stderr.


    # Return

    :return: (tuple(kfold_cv_statistics, kfold_split_indices), dict) a tuple of
            dictionaries `kfold_cv_statistics`: contains metrics from cv run.
             `kfold_split_indices`: indices to split training data into
             training fold and test fold.
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
            (eval_stats, train_stats, preprocessed_data, output_directory) = model.experiment(
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
            train_stats_dict = dataclasses.asdict(train_stats)
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


@PublicAPI
def get_experiment_description(
    config,
    dataset=None,
    training_set=None,
    validation_set=None,
    test_set=None,
    training_set_metadata=None,
    data_format=None,
    backend=None,
    random_seed=None,
):
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

    gpu_info = {}
    if torch.cuda.is_available():
        # Assumption: All nodes are of the same instance type.
        # TODO: fix for Ray where workers may be of different skus
        gpu_info = {"gpu_type": torch.cuda.get_device_name(0), "gpus_per_node": torch.cuda.device_count()}

    compute_description = {"num_nodes": backend.num_nodes, **gpu_info}

    description["compute"] = compute_description

    return description
