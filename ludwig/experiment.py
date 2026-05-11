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
import os
import sys

import pandas as pd

from ludwig.api import kfold_cross_validate, LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import CONTINUE_PROMPT, FULL, HYPEROPT, HYPEROPT_WARNING, TEST, TRAINING, VALIDATION
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_config_from_str, load_yaml, save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig, query_yes_no

logger = logging.getLogger(__name__)


def experiment_cli(
    config: str | dict,
    dataset: str | dict | pd.DataFrame = None,
    training_set: str | dict | pd.DataFrame = None,
    validation_set: str | dict | pd.DataFrame = None,
    test_set: str | dict | pd.DataFrame = None,
    training_set_metadata: str | dict | None = None,
    data_format: str | None = None,
    experiment_name: str = "experiment",
    model_name: str = "run",
    model_load_path: str | None = None,
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
    gpus: str | int | list[int] | None = None,
    gpu_memory_limit: float | None = None,
    allow_parallel_threads: bool = True,
    callbacks: list[Callback] | None = None,
    backend: Backend | str = None,
    random_seed: int = default_random_seed,
    logging_level: int = logging.INFO,
    **kwargs,
):
    """Train a model and evaluate it on a test split, saving both model and statistics.

    Args:
        config: In-memory config dict or path to a YAML config file.
        dataset: Source containing the entire dataset. If it has a split
            column, it will be used for splitting (0: train, 1: validation,
            2: test); otherwise the dataset will be randomly split.
        training_set: Source containing training data.
        validation_set: Source containing validation data.
        test_set: Source containing test data.
        training_set_metadata: Metadata JSON file or loaded metadata dict.
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
        eval_split: Split to evaluate on. Valid values: ``'training'``,
            ``'validation'``, ``'test'``.
        skip_save_training_description: Disable saving the description JSON
            file.
        skip_save_training_statistics: Disable saving training statistics
            JSON file.
        skip_save_model: Disable saving model weights after each epoch the
            validation metric improves.
        skip_save_progress: Disable saving weights and stats after each epoch.
        skip_save_log: Disable saving TensorBoard logs.
        skip_save_processed_input: Disable caching preprocessed input.
        skip_save_unprocessed_output: If ``True``, skip saving raw numpy
            output files; only postprocessed CSV files are saved.
        skip_save_predictions: Disable saving test prediction CSV files.
        skip_save_eval_stats: Disable saving test statistics JSON file.
        skip_collect_predictions: Skip collecting postprocessed predictions
            during evaluation.
        skip_collect_overall_stats: Skip collecting overall stats during
            evaluation.
        output_directory: Directory that will contain all results.
        gpus: List of GPUs available for training.
        gpu_memory_limit: Maximum memory fraction ``[0, 1]`` allowed to
            allocate per GPU device.
        allow_parallel_threads: Allow PyTorch to use multithreading
            parallelism.
        callbacks: List of ``Callback`` objects providing hooks into the
            Ludwig pipeline.
        backend: Backend or string name of the backend to use.
        random_seed: Random seed for weights initialization, splits, and
            shuffling.
        logging_level: Log level sent to stderr.

    Returns:
        Tuple of ``(model, eval_stats, train_stats, preprocessed_data,
        output_directory)`` where ``model`` is the trained ``LudwigModel``,
        ``eval_stats`` are per-split evaluation metrics, ``train_stats``
        are per-epoch training metrics, ``preprocessed_data`` is a tuple of
        ``(training_set, validation_set, test_set)``, and
        ``output_directory`` is the path where results were saved.
    """
    if HYPEROPT in config:
        if not query_yes_no(HYPEROPT_WARNING + CONTINUE_PROMPT):
            exit(1)

    if isinstance(config, str):
        config = load_yaml(config)
    backend = initialize_backend(backend or config.get("backend"))

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
    eval_stats, train_stats, preprocessed_data, output_directory = model.experiment(
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        model_resume_path=model_resume_path,
        eval_split=eval_split,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        skip_collect_predictions=skip_collect_predictions,
        skip_collect_overall_stats=skip_collect_overall_stats,
        output_directory=output_directory,
        random_seed=random_seed,
    )

    return model, eval_stats, train_stats, preprocessed_data, output_directory


def kfold_cross_validate_cli(
    k_fold,
    config=None,
    dataset=None,
    data_format=None,
    output_directory="results",
    random_seed=default_random_seed,
    skip_save_k_fold_split_indices=False,
    **kwargs,
):
    """Run k-fold cross validation and save results to ``output_directory``.

    Args:
        k_fold: Number of folds to create for cross-validation.
        config: Config dict or path to a YAML config file.
        dataset: Dataset source.
        data_format: Format to interpret the dataset.
        output_directory: Directory into which to write results.
        random_seed: Random seed used for k-fold splits.
        skip_save_k_fold_split_indices: If ``True``, skip saving the per-fold
            split index arrays.
    """

    kfold_cv_stats, kfold_split_indices = kfold_cross_validate(
        k_fold,
        config=config,
        dataset=dataset,
        data_format=data_format,
        output_directory=output_directory,
        random_seed=random_seed,
    )

    # save k-fold cv statistics
    save_json(os.path.join(output_directory, "kfold_training_statistics.json"), kfold_cv_stats)

    # save k-fold split indices
    if not skip_save_k_fold_split_indices:
        save_json(os.path.join(output_directory, "kfold_split_indices.json"), kfold_split_indices)


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script trains and evaluates a model", prog="ludwig experiment", usage="%(prog)s [options]"
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
        "-es",
        "--eval_split",
        default=TEST,
        choices=[TRAINING, VALIDATION, TEST, FULL],
        help="the split to evaluate the model on",
    )

    parser.add_argument(
        "-sspi",
        "--skip_save_processed_input",
        help="skips saving intermediate HDF5 and JSON files",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-ssuo",
        "--skip_save_unprocessed_output",
        help="skips saving intermediate NPY output files",
        action="store_true",
        default=False,
    )

    # -----------------
    # K-fold parameters
    # -----------------
    parser.add_argument(
        "-kf", "--k_fold", type=int, default=None, help="number of folds for a k-fold cross validation run "
    )
    parser.add_argument(
        "-skfsi",
        "--skip_save_k_fold_split_indices",
        action="store_true",
        default=False,
        help="disables saving indices generated to split training data set "
        "for the k-fold cross validation run, but if it is not needed "
        "turning it off can slightly increase the overall speed",
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
        "-sstp",
        "--skip_save_predictions",
        help="skips saving test predictions CSV files",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sstes",
        "--skip_save_eval_stats",
        help="skips saving eval statistics JSON file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-ssm",
        "--skip_save_model",
        action="store_true",
        default=False,
        help="disables saving model weights and hyperparameters each time "
        "the model improves. "
        "By default Ludwig saves model weights after each epoch "
        "the validation metric improves, but if the model is really big "
        "that can be time consuming. If you do not want to keep "
        "the weights and just find out what performance a model can get "
        "with a set of hyperparameters, use this parameter to skip it,"
        "but the model will not be loadable later on",
    )
    parser.add_argument(
        "-ssp",
        "--skip_save_progress",
        action="store_true",
        default=False,
        help="disables saving progress each epoch. By default Ludwig saves "
        "weights and stats after each epoch for enabling resuming "
        "of training, but if the model is really big that can be "
        "time consuming and will uses twice as much space, use "
        "this parameter to skip it, but training cannot be resumed "
        "later on",
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
    parser.add_argument("-g", "--gpus", nargs="+", type=int, default=None, help="list of GPUs to use")
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
        callback.on_cmdline("experiment", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.experiment")

    args.backend = initialize_backend(args.backend or args.config.get("backend"))
    if args.backend.is_coordinator():
        print_ludwig("Experiment", LUDWIG_VERSION)

    if args.k_fold is None:
        experiment_cli(**vars(args))
    else:
        kfold_cross_validate_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
