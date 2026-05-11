#! /usr/bin/env python
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
import argparse
import logging
import sys

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import CONTINUE_PROMPT, HYPEROPT, HYPEROPT_WARNING
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig, query_yes_no

logger = logging.getLogger(__name__)


def train_cli(
    config: str | dict | None = None,
    dataset: str | dict | pd.DataFrame = None,
    training_set: str | dict | pd.DataFrame = None,
    validation_set: str | dict | pd.DataFrame = None,
    test_set: str | dict | pd.DataFrame = None,
    training_set_metadata: str | dict | None = None,
    data_format: str | None = None,
    experiment_name: str = "api_experiment",
    model_name: str = "run",
    model_load_path: str | None = None,
    model_resume_path: str | None = None,
    skip_save_training_description: bool = False,
    skip_save_training_statistics: bool = False,
    skip_save_model: bool = False,
    skip_save_progress: bool = False,
    skip_save_log: bool = False,
    skip_save_processed_input: bool = False,
    output_directory: str = "results",
    gpus: str | int | list[int] | None = None,
    gpu_memory_limit: float | None = None,
    allow_parallel_threads: bool = True,
    callbacks: list[Callback] | None = None,
    backend: Backend | str = None,
    random_seed: int = default_random_seed,
    logging_level: int = logging.INFO,
    **kwargs,
) -> None:
    """Build and train a Ludwig model.

    Args:
        config: In-memory config dict or path to a YAML config file.
        dataset: Source containing the entire dataset. If it has a split
            column, it will be used for splitting (0: train, 1: validation,
            2: test); otherwise the dataset will be randomly split.
        training_set: Source containing training data.
        validation_set: Source containing validation data.
        test_set: Source containing test data.
        training_set_metadata: Metadata JSON file or loaded metadata dict.
            Intermediate preprocessed structure containing feature mappings
            created the first time an input file is used.
        data_format: Format to interpret data sources. Inferred automatically
            if not specified. Valid values: ``'auto'``, ``'csv'``,
            ``'excel'``, ``'feather'``, ``'fwf'``, ``'hdf5'``,
            ``'html'``, ``'json'``, ``'jsonl'``, ``'parquet'``,
            ``'pickle'``, ``'sas'``, ``'spss'``, ``'stata'``, ``'tsv'``.
        experiment_name: Name for the experiment.
        model_name: Name of the model being used.
        model_load_path: If specified, load this pre-trained model as
            initialization (useful for transfer learning).
        model_resume_path: Resume training from this checkpoint directory.
            Config, statistics, loss, and optimizer state are all restored.
        skip_save_training_description: Disable saving the description JSON
            file.
        skip_save_training_statistics: Disable saving training statistics
            JSON file.
        skip_save_model: Disable saving model weights after each epoch the
            validation metric improves. The returned model will have weights
            from the final epoch rather than the best epoch.
        skip_save_progress: Disable saving weights and stats after each epoch
            (disables training resumption).
        skip_save_log: Disable saving TensorBoard logs.
        skip_save_processed_input: Disable caching preprocessed input as
            HDF5/JSON files.
        output_directory: Directory that will contain training statistics,
            TensorBoard logs, the saved model, and training progress files.
        gpus: List of GPUs available for training.
        gpu_memory_limit: Maximum memory fraction ``[0, 1]`` allowed to
            allocate per GPU device.
        allow_parallel_threads: Allow PyTorch to use multithreading
            parallelism (improves performance at the cost of determinism).
        callbacks: List of ``Callback`` objects providing hooks into the
            Ludwig pipeline.
        backend: Backend or string name of the backend to use for
            preprocessing and training.
        random_seed: Random seed for weights initialization, splits, and
            shuffling.
        logging_level: Log level sent to stderr.
    """
    if HYPEROPT in config:
        if not query_yes_no(HYPEROPT_WARNING + CONTINUE_PROMPT):
            exit(1)
        # Stop gap: remove hyperopt from the config to prevent interference with training step sizes
        # TODO: https://github.com/ludwig-ai/ludwig/issues/2633
        # Need to investigate why the presence of hyperopt in the config interferes with training step sizes
        config.pop(HYPEROPT)

    if model_load_path:
        model = LudwigModel.load(
            model_load_path,
            logging_level=logging_level,
            backend=backend,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
        )
    else:
        model = LudwigModel(
            config=config,
            logging_level=logging_level,
            backend=backend,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
        )
    model.train(
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
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script trains a model", prog="ludwig train", usage="%(prog)s [options]"
    )

    # ----------------------------
    # Experiment naming parameters
    # ----------------------------
    parser.add_argument("--output_directory", type=str, default="results", help="directory that contains the results")
    parser.add_argument("--experiment_name", type=str, default="experiment", help="experiment name")
    parser.add_argument("--model_name", type=str, default="run", help="name for the model")

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument(
        "--dataset",
        help="input data file path. "
        "If it has a split column, it will be used for splitting "
        "(0: train, 1: validation, 2: test), "
        "otherwise the dataset will be randomly split",
    )
    parser.add_argument("--training_set", help="input train data file path")
    parser.add_argument("--validation_set", help="input validation data file path")
    parser.add_argument("--test_set", help="input test data file path")

    parser.add_argument(
        "--training_set_metadata",
        help="input metadata JSON file path. An intermediate preprocessed file "
        "containing the mappings of the input file created "
        "the first time a file is used, in the same directory "
        "with the same name and a .json extension",
    )

    parser.add_argument(
        "--data_format",
        help="format of the input data",
        default="auto",
        choices=[
            "auto",
            "csv",
            "excel",
            "feather",
            "fwf",
            "hdf5",
            "htmltables",
            "json",
            "jsonl",
            "parquet",
            "pickle",
            "sas",
            "spss",
            "stata",
            "tsv",
        ],
    )

    parser.add_argument(
        "-sspi",
        "--skip_save_processed_input",
        help="skips saving intermediate HDF5 and JSON files",
        action="store_true",
        default=False,
    )

    # ----------------
    # Model parameters
    # ----------------
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument(
        "-c",
        "--config",
        type=load_yaml,
        help="Path to the YAML file containing the model configuration",
    )
    config.add_argument(
        "-cs",
        "--config_str",
        dest="config",
        type=load_config_from_str,
        help="JSON or YAML serialized string of the model configuration",
    )

    parser.add_argument("-mlp", "--model_load_path", help="path of a pretrained model to load as initialization")
    parser.add_argument("-mrp", "--model_resume_path", help="path of the model directory to resume training of")
    parser.add_argument(
        "-sstd",
        "--skip_save_training_description",
        action="store_true",
        default=False,
        help="disables saving the description JSON file",
    )
    parser.add_argument(
        "-ssts",
        "--skip_save_training_statistics",
        action="store_true",
        default=False,
        help="disables saving training statistics JSON file",
    )
    parser.add_argument(
        "-ssm",
        "--skip_save_model",
        action="store_true",
        default=False,
        help="disables saving weights each time the model improves. "
        "By default Ludwig saves  weights after each epoch "
        "the validation metric (improves, but  if the model is really big "
        "that can be time consuming. If you do not want to keep "
        "the weights and just find out what performance a model can get "
        "with a set of hyperparameters, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssp",
        "--skip_save_progress",
        action="store_true",
        default=False,
        help="disables saving weights after each epoch. By default ludwig saves "
        "weights after each epoch for enabling resuming of training, but "
        "if the model is really big that can be time consuming and will "
        "save twice as much space, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssl",
        "--skip_save_log",
        action="store_true",
        default=False,
        help="disables saving TensorBoard logs. By default Ludwig saves "
        "logs for the TensorBoard, but if it is not needed turning it off "
        "can slightly increase the overall speed",
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="a random seed that is going to be used anywhere there is a call "
        "to a random number generator: data splitting, parameter "
        "initialization and training set shuffling",
    )
    parser.add_argument("-g", "--gpus", nargs="+", type=int, default=None, help="list of gpus to use")
    parser.add_argument(
        "-gml",
        "--gpu_memory_limit",
        type=float,
        default=None,
        help="maximum memory fraction [0, 1] allowed to allocate per GPU device",
    )
    parser.add_argument(
        "-dpt",
        "--disable_parallel_threads",
        action="store_false",
        dest="allow_parallel_threads",
        help="disable PyTorch from using multithreading for reproducibility",
    )
    parser.add_argument(
        "-b",
        "--backend",
        help="specifies backend to use for parallel / distributed execution, defaults to local execution",
        choices=ALL_BACKENDS,
    )
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("train", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.train")

    args.backend = initialize_backend(args.backend or args.config.get("backend"))
    if args.backend.is_coordinator():
        print_ludwig("Train", LUDWIG_VERSION)

    train_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
