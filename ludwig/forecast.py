import argparse
import logging
import sys
from typing import List, Optional, Union

import pandas as pd

from ludwig.api import LudwigModel
from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def forecast_cli(
    model_path: str,
    dataset: Union[str, dict, pd.DataFrame] = None,
    data_format: Optional[str] = None,
    horizon: int = 1,
    output_directory: Optional[str] = None,
    output_format: str = "parquet",
    callbacks: List[Callback] = None,
    backend: Union[Backend, str] = None,
    logging_level: int = logging.INFO,
    **kwargs,
) -> None:
    """Loads pre-trained model to forecast on the provided dataset.

    # Inputs

    :param model_path: (str) filepath to pre-trained model.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used in the prediction.
    :param data_format: (str, default: `None`) format to interpret data
        sources. Will be inferred automatically if not specified.
    :param horizon: How many samples into the future to forecast.
    :param output_directory: (str, default: `'results'`) the directory that
        will contain the forecasted values.
    :param output_format: (str) format of the output dataset.
    :param callbacks: (list, default: `None`) a list of
        `ludwig.callbacks.Callback` objects that provide hooks into the
        Ludwig pipeline.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param logging_level: (int) Log level that will be sent to stderr.

    # Returns

    :return: ('None')
    """
    model = LudwigModel.load(
        model_path,
        logging_level=logging_level,
        backend=backend,
        callbacks=callbacks,
    )
    model.forecast(
        dataset=dataset,
        data_format=data_format,
        horizon=horizon,
        output_directory=output_directory,
        output_format=output_format,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script loads a pretrained model and uses it to forecast",
        prog="ludwig forecast",
        usage="%(prog)s [options]",
    )

    parser.add_argument(
        "-n", "--horizon", help="horizon, or number of steps in the future to forecast", type=int, default=1
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
        "-of",
        "--output_format",
        help="format to write the output dataset",
        default="parquet",
        choices=[
            "csv",
            "parquet",
        ],
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
        callback.on_cmdline("forecast", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.forecast")

    args.backend = initialize_backend(args.backend)
    if args.backend.is_coordinator():
        print_ludwig("Forecast", LUDWIG_VERSION)
        logger.info(f"Dataset path: {args.dataset}")
        logger.info(f"Model path: {args.model_path}")
        logger.info("")

    forecast_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
