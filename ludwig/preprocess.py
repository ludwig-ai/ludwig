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
import yaml

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def preprocess_cli(
    preprocessing_config: str | dict | None = None,
    dataset: str | dict | pd.DataFrame = None,
    training_set: str | dict | pd.DataFrame = None,
    validation_set: str | dict | pd.DataFrame = None,
    test_set: str | dict | pd.DataFrame = None,
    training_set_metadata: str | dict | None = None,
    data_format: str | None = None,
    random_seed: int = default_random_seed,
    logging_level: int = logging.INFO,
    callbacks: list[Callback] | None = None,
    backend: Backend | str = None,
    **kwargs,
) -> None:
    """Preprocess a dataset and cache the result to disk.

    Args:
        preprocessing_config: In-memory config dict or path to a YAML config
            file. Only preprocessing settings are used; encoder/decoder/
            combiner/training parameters are ignored.
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
        random_seed: Random seed for splits and any other random function.
        logging_level: Log level sent to stderr.
        callbacks: List of ``Callback`` objects providing hooks into the
            Ludwig pipeline.
        backend: Backend or string name of the backend to use.
    """
    model = LudwigModel(
        config=preprocessing_config,
        logging_level=logging_level,
        callbacks=callbacks,
        backend=backend,
    )
    model.preprocess(
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        skip_save_processed_input=False,
        random_seed=random_seed,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script preprocess a dataset", prog="ludwig preprocess", usage="%(prog)s [options]"
    )

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

    # ----------------
    # Model parameters
    # ----------------
    preprocessing_def = parser.add_mutually_exclusive_group(required=True)
    preprocessing_def.add_argument(
        "-pc",
        "--preprocessing_config",
        dest="preprocessing_config",
        type=load_yaml,
        help="YAML file describing the preprocessing. "
        "Ignores --preprocessing_config."
        "Uses the same format of config, "
        "but ignores encoder specific parameters, "
        "decoder specific parameters, combiner and training parameters",
    )
    preprocessing_def.add_argument(
        "-pcs",
        "--preprocessing_config_str",
        type=yaml.safe_load,
        help="preproceesing config. "
        "Uses the same format of config, "
        "but ignores encoder specific parameters, "
        "decoder specific parameters, combiner and training parameters",
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
        callback.on_cmdline("preprocess", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.preprocess")

    args.backend = initialize_backend(args.backend)
    if args.backend.is_coordinator():
        print_ludwig("Preprocess", LUDWIG_VERSION)

    preprocess_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
