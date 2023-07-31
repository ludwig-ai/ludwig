#! /usr/bin/env python
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
import argparse
import logging
import sys
from typing import List, Optional, Union

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import FULL, TEST, TRAINING, VALIDATION
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def evaluate_cli(
    model_path: str,
    dataset: Union[str, dict, pd.DataFrame] = None,
    data_format: str = None,
    split: str = FULL,
    batch_size: int = 128,
    skip_save_unprocessed_output: bool = False,
    skip_save_predictions: bool = False,
    skip_save_eval_stats: bool = False,
    skip_collect_predictions: bool = False,
    skip_collect_overall_stats: bool = False,
    output_directory: str = "results",
    gpus: Union[str, int, List[int]] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    callbacks: List[Callback] = None,
    backend: Union[Backend, str] = None,
    logging_level: int = logging.INFO,
    **kwargs,
) -> None:
    """Loads pre-trained model and evaluates its performance by comparing the predictions against ground truth.

     # Inputs

     :param model_path: (str) filepath to pre-trained model.
     :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
         source containing the entire dataset to be used in the evaluation.
     :param data_format: (str, default: `None`) format to interpret data
         sources. Will be inferred automatically if not specified.  Valid
         formats are `'auto'`, `'csv'`, `'excel'`, `'feather'`,
         `'fwf'`, `'hdf5'` (cache file produced during previous training),
         `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
         `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
         `'stata'`, `'tsv'`.
     :param split: (str, default: `full`) split on which
         to perform predictions. Valid values are `'training'`, `'validation'`,
         `'test'` and `'full'`.
     :param batch_size: (int, default `128`) size of batches for processing.
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
     :param gpus: (list, default: `None`) list of GPUs that are available
         for training.
     :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
            [0, 1] allowed to allocate per GPU device.
     :param allow_parallel_threads: (bool, default: `True`) allow PyTorch
         to use multithreading parallelism to improve performance at
         the cost of determinism.
     :param callbacks: (list, default: `None`) a list of
         `ludwig.callbacks.Callback` objects that provide hooks into the
         Ludwig pipeline.
     :param backend: (Union[Backend, str]) `Backend` or string name
         of backend to use to execute preprocessing / training steps.
     :param logging_level: (int) Log level that will be sent to stderr.

     # Returns

     :return: (`None`)
    """
    model = LudwigModel.load(
        model_path,
        logging_level=logging_level,
        backend=backend,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        callbacks=callbacks,
    )
    model.evaluate(
        dataset=dataset,
        data_format=data_format,
        batch_size=batch_size,
        split=split,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        collect_predictions=not skip_collect_predictions,
        collect_overall_stats=not skip_collect_overall_stats,
        output_directory=output_directory,
        return_type="dict",
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model "
        "and evaluates its performance by comparing"
        "its predictions with ground truth.",
        prog="ludwig evaluate",
        usage="%(prog)s [options]",
    )

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument("--dataset", help="input data file path", required=True)
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
            "html" "tables",
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
        "-s", "--split", default=FULL, choices=[TRAINING, VALIDATION, TEST, FULL], help="the split to test the model on"
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument("-m", "--model_path", help="model to load", required=True)

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        "-od", "--output_directory", type=str, default="results", help="directory that contains the results"
    )
    parser.add_argument(
        "-ssuo",
        "--skip_save_unprocessed_output",
        help="skips saving intermediate NPY output files",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sses",
        "--skip_save_eval_stats",
        help="skips saving intermediate JSON eval statistics",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-scp", "--skip_collect_predictions", help="skips collecting predictions", action="store_true", default=False
    )
    parser.add_argument(
        "-scos",
        "--skip_collect_overall_stats",
        help="skips collecting overall stats",
        action="store_true",
        default=False,
    )

    # ------------------
    # Generic parameters
    # ------------------
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="size of batches")

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument("-g", "--gpus", type=int, default=0, help="list of gpu to use")
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
        help="specifies backend to use for parallel / distributed execution, "
        "defaults to local execution or Horovod if called using horovodrun",
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
    args.evaluate_performance = True

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("evaluate", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.test_performance")

    backend = initialize_backend(args.backend)
    if backend.is_coordinator():
        print_ludwig("Evaluate", LUDWIG_VERSION)
        logger.info(f"Dataset path: {args.dataset}")
        logger.info(f"Model path: {args.model_path}")
        logger.info("")

    evaluate_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
