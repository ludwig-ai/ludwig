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
import os
import sys
from typing import List, Optional, Union

import numpy as np
import torchinfo

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend
from ludwig.callbacks import Callback
from ludwig.constants import FULL, TEST, TRAINING, VALIDATION
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_boxed, print_ludwig
from ludwig.utils.strings_utils import make_safe_filename

logger = logging.getLogger(__name__)


def collect_activations(
    model_path: str,
    layers: List[str],
    dataset: str,
    data_format: str = None,
    split: str = FULL,
    batch_size: int = 128,
    output_directory: str = "results",
    gpus: List[str] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    callbacks: List[Callback] = None,
    backend: Union[Backend, str] = None,
    **kwargs,
) -> List[str]:
    """Uses the pretrained model to collect the tensors corresponding to a datapoint in the dataset. Saves the
    tensors to the experiment directory.

    # Inputs

    :param model_path: (str) filepath to pre-trained model.
    :param layers: (List[str]) list of strings for layer names in the model
        to collect activations.
    :param dataset: (str) source
        containing the data to make predictions.
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

    # Return

    :return: (List[str]) list of filepath to `*.npy` files containing
        the activations.
    """
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_directory}")
    logger.info("\n")

    model = LudwigModel.load(
        model_path,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        callbacks=callbacks,
        backend=backend,
    )

    # collect activations
    print_boxed("COLLECT ACTIVATIONS")
    collected_tensors = model.collect_activations(
        layers, dataset, data_format=data_format, split=split, batch_size=batch_size
    )

    # saving
    os.makedirs(output_directory, exist_ok=True)
    saved_filenames = save_tensors(collected_tensors, output_directory)

    logger.info(f"Saved to: {output_directory}")
    return saved_filenames


def collect_weights(model_path: str, tensors: List[str], output_directory: str = "results", **kwargs) -> List[str]:
    """Loads a pretrained model and collects weights.

    # Inputs
    :param model_path: (str) filepath to pre-trained model.
    :param tensors: (list, default: `None`) List of tensor names to collect
        weights
    :param output_directory: (str, default: `'results'`) the directory where
        collected weights will be stored.

    # Return

    :return: (List[str]) list of filepath to `*.npy` files containing
        the weights.
    """
    logger.info(f"Model path: {model_path}")
    logger.info(f"Output path: {output_directory}")
    logger.info("\n")

    model = LudwigModel.load(model_path)

    # collect weights
    print_boxed("COLLECT WEIGHTS")
    collected_tensors = model.collect_weights(tensors)

    # saving
    os.makedirs(output_directory, exist_ok=True)
    saved_filenames = save_tensors(collected_tensors, output_directory)

    logger.info(f"Saved to: {output_directory}")
    return saved_filenames


def save_tensors(collected_tensors, output_directory):
    filenames = []
    for tensor_name, tensor_value in collected_tensors:
        np_filename = os.path.join(output_directory, make_safe_filename(tensor_name) + ".npy")
        np.save(np_filename, tensor_value.detach().cpu().numpy())
        filenames.append(np_filename)
    return filenames


def print_model_summary(model_path: str, **kwargs) -> None:
    """Loads a pretrained model and prints names of weights and layers activations.

    # Inputs
    :param model_path: (str) filepath to pre-trained model.

    # Return
    :return: (`None`)
    """
    model = LudwigModel.load(model_path)
    # Model's dict inputs are wrapped in a list, required by torchinfo.
    logger.info(torchinfo.summary(model.model, input_data=[model.model.get_model_inputs()], depth=20))

    logger.info("\nModules:\n")
    for name, _ in model.model.named_children():
        logger.info(name)

    logger.info("\nParameters:\n")
    for name, _ in model.model.named_parameters():
        logger.info(name)


def cli_collect_activations(sys_argv):
    """Command Line Interface to communicate with the collection of tensors and there are several options that can
    specified when calling this function:

    --data_csv: Filepath for the input csv
    --data_hdf5: Filepath for the input hdf5 file, if there is a csv file, this
                 is not read
    --d: Refers to the dataset type of the file being read, by default is
         *generic*
    --s: Refers to the split of the data, can be one of: train, test,
         validation, full
    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --t: Tensors to collect
    --od: Output directory of the model, defaults to results
    --bs: Batch size
    --g: Number of gpus that are to be used
    --gf: Fraction of each GPUs memory to use.
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model and uses it collect "
        "tensors for each datapoint in the dataset.",
        prog="ludwig collect_activations",
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
        "-s",
        "--split",
        default=FULL,
        choices=[TRAINING, VALIDATION, TEST, FULL],
        help="the split to obtain the model activations from",
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument("-m", "--model_path", help="model to load", required=True)
    parser.add_argument("-lyr", "--layers", help="tensors to collect", nargs="+", required=True)

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        "-od", "--output_directory", type=str, default="results", help="directory that contains the results"
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

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("collect_activations", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.collect")

    print_ludwig("Collect Activations", LUDWIG_VERSION)

    collect_activations(**vars(args))


def cli_collect_weights(sys_argv):
    """Command Line Interface to collecting the weights for the model.

    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --t: Tensors to collect
    --od: Output directory of the model, defaults to results
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model " "and uses it collect weights.",
        prog="ludwig collect_weights",
        usage="%(prog)s [options]",
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument("-m", "--model_path", help="model to load", required=True)
    parser.add_argument("-t", "--tensors", help="tensors to collect", nargs="+", required=True)

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        "-od", "--output_directory", type=str, default="results", help="directory that contains the results"
    )

    # ------------------
    # Runtime parameters
    # ------------------
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
        callback.on_cmdline("collect_weights", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.collect")

    print_ludwig("Collect Weights", LUDWIG_VERSION)

    collect_weights(**vars(args))


def cli_collect_summary(sys_argv):
    """Command Line Interface to collecting a summary of the model layers and weights.

    --m: Input model that is necessary to collect to the tensors, this is a
         required *option*
    --v: Verbose: Defines the logging level that the user will be exposed to
    """
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model "
        "and prints names of weights and layers activations "
        "to use with other collect commands",
        prog="ludwig collect_summary",
        usage="%(prog)s [options]",
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument("-m", "--model_path", help="model to load", required=True)

    # ------------------
    # Runtime parameters
    # ------------------
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
        callback.on_cmdline("collect_summary", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.collect")

    print_ludwig("Collect Summary", LUDWIG_VERSION)

    print_model_summary(**vars(args))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "activations":
            cli_collect_activations(sys.argv[2:])
        elif sys.argv[1] == "weights":
            cli_collect_weights(sys.argv[2:])
        elif sys.argv[1] == "names":
            cli_collect_summary(sys.argv[2:])
        else:
            print("Unrecognized command")
    else:
        print("Unrecognized command")
