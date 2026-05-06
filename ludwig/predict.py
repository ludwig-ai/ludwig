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
from ast import literal_eval

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.constants import FULL, TEST, TRAINING, VALIDATION
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def predict_cli(
    model_path: str,
    dataset: str | dict | pd.DataFrame = None,
    data_format: str | None = None,
    split: str = FULL,
    batch_size: int = 128,
    generation_config: str | None = None,
    skip_save_unprocessed_output: bool = False,
    skip_save_predictions: bool = False,
    output_directory: str = "results",
    gpus: str | int | list[int] | None = None,
    gpu_memory_limit: float | None = None,
    allow_parallel_threads: bool = True,
    callbacks: list[Callback] | None = None,
    backend: Backend | str = None,
    logging_level: int = logging.INFO,
    **kwargs,
) -> None:
    """Load a pre-trained model and generate predictions on the provided dataset.

    Args:
        model_path: Filepath to the pre-trained model directory.
        dataset: Source containing the dataset to predict on.
        data_format: Format to interpret data sources. Inferred automatically
            if not specified. Valid values: ``'auto'``, ``'csv'``,
            ``'excel'``, ``'feather'``, ``'fwf'``, ``'hdf5'``,
            ``'html'``, ``'json'``, ``'jsonl'``, ``'parquet'``,
            ``'pickle'``, ``'sas'``, ``'spss'``, ``'stata'``, ``'tsv'``.
        split: Split to perform predictions on. Valid values:
            ``'training'``, ``'validation'``, ``'test'``, ``'full'``.
        batch_size: Number of samples per prediction batch.
        generation_config: JSON-formatted string of generation parameters
            for LLM predictions (merged with the model's generation config).
        skip_save_unprocessed_output: If ``True``, skip saving raw numpy
            output files; only postprocessed CSV files are saved.
        skip_save_predictions: If ``True``, skip saving prediction CSV files.
        output_directory: Directory that will contain prediction results.
        gpus: List of GPUs available for inference.
        gpu_memory_limit: Maximum memory fraction ``[0, 1]`` allowed to
            allocate per GPU device.
        allow_parallel_threads: Allow PyTorch to use multithreading
            parallelism (improves performance at the cost of determinism).
        callbacks: List of ``Callback`` objects providing hooks into the
            Ludwig pipeline.
        backend: Backend or string name of the backend to use.
        logging_level: Log level sent to stderr.
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
    model.predict(
        dataset=dataset,
        data_format=data_format,
        split=split,
        batch_size=batch_size,
        generation_config=literal_eval(generation_config) if generation_config else None,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        output_directory=output_directory,
        return_type="dict",
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model and uses it to predict",
        prog="ludwig predict",
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
            "html",
            "tables",
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
    parser.add_argument("-gc", "--generation_config", help="generation config (LLMs only)", default=None)

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
        "-sstp",
        "--skip_save_predictions",
        help="skips saving predictions CSV files",
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
        callback.on_cmdline("predict", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.predict")

    args.backend = initialize_backend(args.backend)
    if args.backend.is_coordinator():
        print_ludwig("Predict", LUDWIG_VERSION)
        logger.info(f"Dataset path: {args.dataset}")
        logger.info(f"Model path: {args.model_path}")
        logger.info("")

    predict_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
