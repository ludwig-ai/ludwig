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

from ludwig.api import LudwigModel
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def export_mlflow(model_path, output_path="mlflow", registered_model_name=None, **kwargs):
    """Exports a trained Ludwig model as an MLflow model.

    # Inputs
    :param model_path: (str) filepath to the trained Ludwig model.
    :param output_path: (str) output directory for the MLflow model.
    :param registered_model_name: (str, default: `None`) register model with this name.
    """
    logger.info(f"Loading Ludwig model from {model_path}")
    model = LudwigModel.load(model_path)
    kwargs = dict(model_path=model_path, output_path=output_path, registered_model_name=registered_model_name)
    model.callback(lambda c: c.on_cmdline("export_mlflow", **kwargs))

    from ludwig.contribs.mlflow.model import export_model

    export_model(model_path, output_path, registered_model_name)


def export_model(model_path, output_path, format="safetensors", **kwargs):
    """Exports a trained Ludwig model in various formats.

    # Inputs
    :param model_path: (str) filepath to the trained Ludwig model.
    :param output_path: (str) output directory for the exported model.
    :param format: (str) export format: safetensors, torch_export, onnx.
    """
    logger.info(f"Loading Ludwig model from {model_path}")
    model = LudwigModel.load(model_path)
    model.export_model(output_path, format=format)


def cli_export_mlflow(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script exports a trained Ludwig model to MLflow format",
        prog="ludwig export_mlflow",
        usage="%(prog)s [options]",
    )
    parser.add_argument("-m", "--model_path", help="path to the trained model", required=True)
    parser.add_argument("-o", "--output_path", type=str, default="mlflow", help="output path")
    parser.add_argument("-rmn", "--registered_model_name", type=str, default=None, help="registered model name")
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="logging level",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )
    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)
    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    print_ludwig("Export MLflow", LUDWIG_VERSION)
    export_mlflow(**vars(args))


def cli_export_model(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script exports a trained Ludwig model to various formats (safetensors, torch_export, onnx)",
        prog="ludwig export_model",
        usage="%(prog)s [options]",
    )
    parser.add_argument("-m", "--model_path", help="path to the trained model", required=True)
    parser.add_argument("-o", "--output_path", type=str, default="exported_model", help="output path")
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="safetensors",
        choices=["safetensors", "torch_export", "onnx"],
        help="export format",
    )
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="logging level",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )
    args = parser.parse_args(sys_argv)
    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    print_ludwig("Export Model", LUDWIG_VERSION)
    export_model(**vars(args))


def cli(sys_argv):
    sub = sys.argv[1] if len(sys.argv) > 1 else None
    if sub == "mlflow":
        cli_export_mlflow(sys.argv[2:])
    elif sub == "model":
        cli_export_model(sys.argv[2:])
    else:
        print(f"Unknown export subcommand: {sub}")
        print("Available: mlflow, model")
        sys.exit(1)


if __name__ == "__main__":
    cli(sys.argv)
